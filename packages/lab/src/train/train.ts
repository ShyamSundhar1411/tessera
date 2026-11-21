import * as tf from "@tensorflow/tfjs";
import { analyze, type Model } from "tessera-engine";
import type { Dataset } from "./datasets";
import { band, evaluate, metricOf, RUN_METRICS, stats, type Evaluation, type RunMetric, type Stats } from "./metrics";
import { buildTf, trainableParams, type TfBuild } from "./toTf";

// Trains several models on the same data, the same way, once per seed, so the
// differences in their curves come from the architecture rather than luck.

export interface TrainOptions {
  epochs: number;
  batchSize: number;
  learningRate: number;
  seed: number;
  /** Run every model once per seed; defaults to [seed]. More seeds show how much of a gap is noise. */
  seeds?: number[];
  /** Report a batch event this many times per epoch (default 20). */
  reportsPerEpoch?: number;
  /** Work out per-layer gradient norms once per epoch (default true). Costs one extra forward and backward pass. */
  gradientNorms?: boolean;
  /** Backends to try in order (default webgl, wasm, cpu). wasm is used only if the app registered it. */
  backends?: string[];
}

export interface TrainEvent {
  /** "batch" during an epoch, "epoch" once it ends and the test split has been evaluated. */
  type: "batch" | "epoch";
  modelIndex: number;
  seed: number;
  /** Which run of this model, the index of its seed. */
  run: number;
  /** Counted from 1. */
  epoch: number;
  /** Batches done in this epoch, counted from 1. */
  batch: number;
  batches: number;
  /** Training loss and accuracy averaged over the epoch so far. */
  loss: number;
  accuracy: number;
  valLoss?: number;
  valAccuracy?: number;
  /** Seconds since this run started. */
  secondsElapsed: number;
  /** The full test-split evaluation, on epoch events. */
  evaluation?: Evaluation;
  /** Gradient norm per engine node id, on epoch events when enabled. Shows which layers are learning. */
  gradientNorms?: Record<string, number>;
}

export interface EpochRecord {
  epoch: number;
  loss: number;
  accuracy: number;
  evaluation: Evaluation;
  gradientNorms?: Record<string, number>;
  secondsElapsed: number;
}

export interface RunResult {
  seed: number;
  epochs: EpochRecord[];
  /** The evaluation after the last completed epoch; null if stopped before one finished. */
  final: Evaluation | null;
  seconds: number;
  aborted: boolean;
}

export type Curve = "loss" | "accuracy" | "valLoss" | "valAccuracy";

export interface ModelResult {
  modelIndex: number;
  name: string;
  params: number;
  runs: RunResult[];
  /** Final test metrics across the runs that finished at least one epoch. */
  summary: Record<RunMetric, Stats>;
  /** Mean and standard deviation per epoch across runs, for a band around the curve. */
  curves: Record<Curve, { mean: number[]; std: number[] }>;
}

export interface TrainResult {
  models: ModelResult[];
  backend: string;
  aborted: boolean;
}

export class DatasetMismatchError extends Error {}

const times = (shape: number[]) => shape.join("×");

/**
 * Says why a model cannot train on a dataset, in words the UI can show, or
 * returns null when it can.
 */
export function checkCompatibility(model: Model, dataset: Pick<Dataset, "label" | "shape" | "classes">): string | null {
  const analysis = analyze(model);
  if (analysis.errors.length) return `${model.name} has a problem: ${analysis.errors[0].message}`;
  const input = model.nodes.find((node) => node.op === "input");
  const inputShape = input ? analysis.info.get(input.id)!.shape : [];
  if (inputShape.join() !== dataset.shape.join()) {
    const what = inputShape.length === 3 ? "images" : "inputs";
    return `${model.name} expects ${times(inputShape)} ${what}, ${dataset.label} is ${times(dataset.shape)}`;
  }
  const out = analysis.outputShape ?? [];
  if (out.length !== 1 || out[0] !== dataset.classes.length)
    return `${model.name} ends in [${out.join(", ")}] outputs, ${dataset.label} has ${dataset.classes.length} classes; make the last layer output ${dataset.classes.length}`;
  return null;
}

/** Picks the first backend that works. In a browser that is usually WebGL. */
export async function initBackend(preferred = ["webgl", "wasm", "cpu"]): Promise<string> {
  for (const name of preferred) {
    if (!tf.findBackendFactory(name)) continue;
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      // Not available here (no WebGL, wasm files missing); try the next one.
    }
  }
  throw new Error(`none of the tf.js backends ${preferred.join(", ")} could start`);
}

export async function trainCompare(
  models: Model[],
  dataset: Dataset,
  options: TrainOptions,
  onProgress: (event: TrainEvent) => void,
  signal?: AbortSignal,
): Promise<TrainResult> {
  for (const model of models) {
    const problem = checkCompatibility(model, dataset);
    if (problem) throw new DatasetMismatchError(problem);
  }
  const backend = await initBackend(options.backends);
  const seeds = options.seeds?.length ? options.seeds : [options.seed];
  const runs: RunResult[][] = models.map(() => []);
  const params: number[] = models.map(() => 0);
  let aborted = false;

  // Seeds on the outside, so after the first pass every model has one run to compare.
  outer: for (let run = 0; run < seeds.length; run++) {
    for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
      const result = await trainRun(models[modelIndex], modelIndex, run, seeds[run], dataset, options, onProgress, signal, (count) => (params[modelIndex] = count));
      runs[modelIndex].push(result);
      if (result.aborted) {
        aborted = true;
        break outer;
      }
    }
  }

  return {
    backend,
    aborted,
    models: models.map((model, modelIndex) => summarise(model.name, modelIndex, params[modelIndex], runs[modelIndex])),
  };
}

export function summarise(name: string, modelIndex: number, params: number, runs: RunResult[]): ModelResult {
  const finished = runs.filter((run) => run.final);
  const summary = Object.fromEntries(RUN_METRICS.map((metric) => [metric, stats(finished.map((run) => metricOf(run.final!, metric)))])) as Record<RunMetric, Stats>;
  const curve = (pick: (epoch: EpochRecord) => number) => band(runs.map((run) => run.epochs.map(pick)));
  return {
    modelIndex,
    name,
    params,
    runs,
    summary,
    curves: {
      loss: curve((e) => e.loss),
      accuracy: curve((e) => e.accuracy),
      valLoss: curve((e) => e.evaluation.loss),
      valAccuracy: curve((e) => e.evaluation.accuracy),
    },
  };
}

async function trainRun(
  model: Model,
  modelIndex: number,
  run: number,
  seed: number,
  dataset: Dataset,
  options: TrainOptions,
  onProgress: (event: TrainEvent) => void,
  signal: AbortSignal | undefined,
  reportParams: (count: number) => void,
): Promise<RunResult> {
  const started = performance.now();
  const elapsed = () => (performance.now() - started) / 1000;
  const built = buildTf(model, { seed });
  reportParams(trainableParams(built.model));
  const optimizer = tf.train.adam(options.learningRate);
  const variables = built.model.trainableWeights.map((w) => w.read() as tf.Variable);
  const random = mulberry32(seed);
  const n = dataset.xTrain.shape[0];
  const batches = Math.ceil(n / options.batchSize);
  const reportEvery = Math.max(1, Math.floor(batches / (options.reportsPerEpoch ?? 20)));
  const epochs: EpochRecord[] = [];
  const result = (aborted: boolean): RunResult => ({ seed, epochs, final: epochs.at(-1)?.evaluation ?? null, seconds: elapsed(), aborted });

  try {
    for (let epoch = 1; epoch <= options.epochs; epoch++) {
      const order = shuffled(n, random);
      let lossSum = 0;
      let correctSum = 0;
      let seen = 0;
      for (let b = 0; b < batches; b++) {
        if (signal?.aborted) return result(true);
        const indices = order.subarray(b * options.batchSize, Math.min(n, (b + 1) * options.batchSize));
        const [lossValue, correct] = trainStep(built, optimizer, variables, dataset, indices);
        const [loss, hits] = await Promise.all([lossValue.data(), correct.data()]);
        tf.dispose([lossValue, correct]);
        lossSum += loss[0] * indices.length;
        correctSum += hits[0];
        seen += indices.length;
        if ((b + 1) % reportEvery === 0 || b === batches - 1) {
          onProgress({ type: "batch", modelIndex, seed, run, epoch, batch: b + 1, batches, loss: lossSum / seen, accuracy: correctSum / seen, secondsElapsed: elapsed() });
          await yieldToBrowser();
        }
      }
      const gradientNorms = options.gradientNorms === false ? undefined : layerGradientNorms(built, variables, dataset, order.subarray(0, Math.min(n, options.batchSize)));
      const evaluation = await evaluateModel(built, dataset, options.batchSize);
      const record: EpochRecord = { epoch, loss: lossSum / seen, accuracy: correctSum / seen, evaluation, gradientNorms, secondsElapsed: elapsed() };
      epochs.push(record);
      onProgress({
        type: "epoch",
        modelIndex,
        seed,
        run,
        epoch,
        batch: batches,
        batches,
        loss: record.loss,
        accuracy: record.accuracy,
        valLoss: evaluation.loss,
        valAccuracy: evaluation.accuracy,
        secondsElapsed: record.secondsElapsed,
        evaluation,
        gradientNorms,
      });
      await yieldToBrowser();
    }
    return result(false);
  } finally {
    optimizer.dispose();
    built.model.dispose();
  }
}

/** The loss for a batch, from logits unless the model ends in its own softmax. */
function lossOf(built: TfBuild, labels: tf.Tensor, output: tf.Tensor): tf.Scalar {
  if (built.endsInSoftmax) return tf.mean(tf.metrics.categoricalCrossentropy(labels, tf.clipByValue(output, 1e-7, 1))) as tf.Scalar;
  return tf.losses.softmaxCrossEntropy(labels, output) as tf.Scalar;
}

function trainStep(built: TfBuild, optimizer: tf.Optimizer, variables: tf.Variable[], dataset: Dataset, indices: Int32Array): [tf.Scalar, tf.Scalar] {
  let correct: tf.Scalar | undefined;
  const loss = tf.tidy(() => {
    const index = tf.tensor1d(indices, "int32");
    const x = tf.gather(dataset.xTrain, index);
    const y = tf.gather(dataset.yTrain, index);
    return optimizer.minimize(
      () => {
        const output = built.model.apply(x, { training: true }) as tf.Tensor;
        // Kept out of the tidy so accuracy needs no second forward pass.
        correct = tf.keep(tf.sum(tf.cast(tf.equal(tf.argMax(output, -1), tf.argMax(y, -1)), "float32")) as tf.Scalar);
        return lossOf(built, y, output);
      },
      true,
      variables,
    )!;
  });
  return [loss, correct!];
}

/** One backward pass on a batch, gradient norm summed over each node's weights (and every copy of a repeated block). */
function layerGradientNorms(built: TfBuild, variables: tf.Variable[], dataset: Dataset, indices: Int32Array): Record<string, number> {
  const nodeOfVariable = new Map<string, string>();
  for (const layer of built.model.layers) {
    const node = built.layerToNode.get(layer.name);
    if (!node) continue;
    for (const weight of layer.trainableWeights) nodeOfVariable.set((weight.read() as tf.Variable).name, node);
  }
  const squares = tf.tidy(() => {
    const index = tf.tensor1d(indices, "int32");
    const x = tf.gather(dataset.xTrain, index);
    const y = tf.gather(dataset.yTrain, index);
    const { grads } = tf.variableGrads(() => lossOf(built, y, built.model.apply(x, { training: true }) as tf.Tensor), variables);
    const names = Object.keys(grads);
    const sums = tf.stack(names.map((name) => tf.sum(tf.square(grads[name]))));
    return { names, values: sums.dataSync() };
  });
  const totals: Record<string, number> = {};
  squares.names.forEach((name, i) => {
    const node = nodeOfVariable.get(name);
    if (node) totals[node] = (totals[node] ?? 0) + squares.values[i];
  });
  for (const node of Object.keys(totals)) totals[node] = Math.sqrt(totals[node]);
  return totals;
}

/** Loss, accuracy, top-5, per-class scores and the confusion matrix on the test split. */
export async function evaluateModel(built: TfBuild, dataset: Dataset, batchSize: number): Promise<Evaluation> {
  const n = dataset.xTest.shape[0];
  const classes = dataset.classes.length;
  const predicted = new Int32Array(n);
  let lossSum = 0;
  let top5Hits = 0;
  for (let start = 0; start < n; start += batchSize) {
    const size = Math.min(batchSize, n - start);
    const [loss, argmax, top5] = tf.tidy(() => {
      const x = dataset.xTest.slice([start, 0, 0, 0], [size, -1, -1, -1]);
      const y = dataset.yTest.slice([start, 0], [size, -1]);
      const output = built.model.apply(x, { training: false }) as tf.Tensor;
      const truth = tf.argMax(y, -1);
      const k = Math.min(5, classes);
      const inTop = tf.sum(tf.cast(tf.equal(tf.topk(output, k).indices, tf.expandDims(truth, 1)), "float32"));
      return [tf.mul(lossOf(built, y, output), size), tf.argMax(output, -1), inTop];
    });
    const [l, a, t] = await Promise.all([loss.data(), argmax.data(), top5.data()]);
    tf.dispose([loss, argmax, top5]);
    lossSum += l[0];
    top5Hits += t[0];
    predicted.set(a as Int32Array, start);
  }
  return evaluate({ truth: dataset.testLabels, predicted, top5Hits, loss: lossSum / n, classes });
}

const yieldToBrowser = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/** A small seeded random number generator, so data order is the same for every model on a seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(n: number, random: () => number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
