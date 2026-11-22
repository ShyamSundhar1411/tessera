import * as tf from "@tensorflow/tfjs";
import type { Model } from "tessera-engine";
import type { Dataset } from "./datasets";
import { buildGraph, type GraphModel } from "./graph";
import { band, evaluate, stats, type Evaluation, type RunSummary } from "./metrics";
import { loadTaskData, planTraining, type InputRole, type PrepareOptions, type PreparedModel, type TaskData, type TaskKind, type TaskSpec, type TokenSplit } from "./tasks";
import { DatasetMismatchError, initBackend, mulberry32, summarise, type EpochRecord, type ModelResult, type RunResult, type TrainEvent, type TrainOptions } from "./train";

// Trains any two (or more) architectures against each other in the browser:
// decoders, encoders, encoder-decoders, vision transformers and CNNs. The task
// comes from the wiring (see tasks.ts), every model runs on every seed with
// the same data in the same order, and the results have the same shape as
// trainCompare's, plus perplexity where predictions are per token.

export class TaskMismatchError extends DatasetMismatchError {}

export interface AnyTrainOptions extends TrainOptions {
  /** How transformers are shrunk and which task variant to use. */
  prepare?: PrepareOptions;
  /** Image tasks: the dataset loaded with loadDataset (prepareForTraining says which). */
  imageDataset?: Dataset;
  /** Character tasks: the text to learn; tiny Shakespeare is fetched when absent. */
  text?: string;
  /** Training examples per epoch for token tasks (default 2048). Image tasks use the whole dataset passed in. */
  trainExamples?: number;
  /** Test examples for token tasks (default 256). */
  testExamples?: number;
  /** Seeds the data (window positions, masks, generated sequences). Default 1234, the same for every model and run. */
  dataSeed?: number;
  /** Clip the global gradient norm to this (default 1; 0 turns it off). Keeps post-norm transformers steady. */
  clipNorm?: number;
  /**
   * Receives each trained model when its run ends, so the page can explain
   * what it learned. The receiver owns the graph and must dispose it; without
   * this option graphs are disposed.
   */
  keepTrained?: (trained: { modelIndex: number; run: number; seed: number; graph: GraphModel; prepared: PreparedModel }) => void;
}

export interface AnyTrainEvent extends TrainEvent {
  task: TaskKind;
  /** exp(training loss), for per-token tasks. */
  perplexity?: number;
  /** exp(test loss), on epoch events of per-token tasks. */
  valPerplexity?: number;
}

export interface AnyModelResult extends ModelResult {
  summary: RunSummary;
  curves: ModelResult["curves"] & { valPerplexity?: { mean: number[]; std: number[] } };
  /** What was shrunk or adapted, and the task. */
  description: string;
  /** The model's parameters before shrinking; `params` is what trained. */
  originalParams: number;
}

export interface AnyTrainResult {
  task: TaskSpec;
  models: AnyModelResult[];
  backend: string;
  aborted: boolean;
}

/**
 * Says whether a set of models can be trained against each other, and on what.
 * Use it before starting to show the task and the shrunk sizes, or the reason.
 */
export function planCompare(models: Model[], prepare?: PrepareOptions) {
  return planTraining(models, prepare);
}

export async function trainCompareAny(models: Model[], options: AnyTrainOptions, onProgress: (event: AnyTrainEvent) => void, signal?: AbortSignal): Promise<AnyTrainResult> {
  const plan = planTraining(models, options.prepare);
  if (!plan.ok) throw new TaskMismatchError(plan.reason);
  const { task, prepared } = plan;
  if (task.kind === "image-classification") {
    const data = options.imageDataset;
    if (!data) throw new TaskMismatchError(`"${task.label}" needs the image dataset: load "${task.imageDataset}" with loadDataset and pass it as imageDataset`);
    const shape = prepared[0].model.nodes.find((node) => node.op === "input")!.args.shape;
    if (data.shape.join("x") !== String(shape)) throw new TaskMismatchError(`the models were fitted to ${String(shape).replaceAll("x", "×")} images, ${data.label} is ${data.shape.join("×")}`);
  }
  const backend = await initBackend(options.backends);
  const data = await loadTaskData(task, { text: options.text, trainExamples: options.trainExamples, testExamples: options.testExamples, dataSeed: options.dataSeed, image: options.imageDataset });
  const seeds = options.seeds?.length ? options.seeds : [options.seed];
  const runs: RunResult[][] = models.map(() => []);
  const params = prepared.map((p) => p.params);
  let aborted = false;

  // Seeds on the outside, so after the first pass every model has one run to compare.
  outer: for (let run = 0; run < seeds.length; run++) {
    for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
      const result = await trainRun(prepared[modelIndex], modelIndex, run, seeds[run], data, options, onProgress, signal);
      runs[modelIndex].push(result);
      if (result.aborted) {
        aborted = true;
        break outer;
      }
    }
  }

  return {
    task,
    backend,
    aborted,
    models: prepared.map((p, modelIndex) => {
      const base = summarise(p.model.name, modelIndex, params[modelIndex], runs[modelIndex]);
      const result: AnyModelResult = { ...base, description: p.description, originalParams: p.originalParams };
      if (task.tokenLevel) {
        const finished = runs[modelIndex].filter((r) => r.final);
        result.summary = { ...base.summary, perplexity: stats(finished.map((r) => r.final!.perplexity ?? NaN)) };
        result.curves = { ...base.curves, valPerplexity: band(runs[modelIndex].map((r) => r.epochs.map((e) => e.evaluation.perplexity ?? NaN))) };
      }
      return result;
    }),
  };
}

// -- One run -----------------------------------------------------------------------

interface Batch {
  feeds: Record<string, tf.Tensor>;
  /** Class ids: [B, T] for token tasks, [B] for classification. */
  targets: tf.Tensor;
  /** Which target positions count, same shape as targets; absent when all do. */
  weights?: tf.Tensor;
}

async function trainRun(
  prepared: PreparedModel,
  modelIndex: number,
  run: number,
  seed: number,
  data: TaskData,
  options: AnyTrainOptions,
  onProgress: (event: AnyTrainEvent) => void,
  signal: AbortSignal | undefined,
): Promise<RunResult> {
  const started = performance.now();
  const elapsed = () => (performance.now() - started) / 1000;
  const graph = buildGraph(prepared.model, { seed });
  const optimizer = tf.train.adam(options.learningRate);
  // The data order depends only on the seed, so every model sees the same batches.
  const random = mulberry32(seed);
  const n = data.image ? data.image.xTrain.shape[0] : data.train!.size;
  const batches = Math.ceil(n / options.batchSize);
  const reportEvery = Math.max(1, Math.floor(batches / (options.reportsPerEpoch ?? 20)));
  const clip = options.clipNorm ?? 1;
  const tokenLevel = data.task.tokenLevel;
  const kind = data.task.kind;
  const epochs: EpochRecord[] = [];
  const result = (aborted: boolean): RunResult => ({ seed, epochs, final: epochs.at(-1)?.evaluation ?? null, seconds: elapsed(), aborted });
  let step = 0;

  try {
    for (let epoch = 1; epoch <= options.epochs; epoch++) {
      const order = shuffled(n, random);
      let lossSum = 0;
      let correctSum = 0;
      let countSum = 0;
      let gradientNorms: Record<string, number> | undefined;
      for (let b = 0; b < batches; b++) {
        if (signal?.aborted) return result(true);
        const indices = order.subarray(b * options.batchSize, Math.min(n, (b + 1) * options.batchSize));
        const wantNorms = options.gradientNorms !== false && b === batches - 1;
        const out = trainStep(graph, optimizer, prepared, data, indices, ++step, clip, wantNorms);
        const [loss, correct, counted] = await Promise.all([out.loss.data(), out.correct.data(), out.count.data()]);
        tf.dispose([out.loss, out.correct, out.count]);
        if (out.norms) gradientNorms = out.norms;
        lossSum += loss[0] * counted[0];
        correctSum += correct[0];
        countSum += counted[0];
        if ((b + 1) % reportEvery === 0 || b === batches - 1) {
          const meanLoss = lossSum / countSum;
          onProgress({ type: "batch", task: kind, modelIndex, seed, run, epoch, batch: b + 1, batches, loss: meanLoss, accuracy: correctSum / countSum, perplexity: tokenLevel ? Math.exp(meanLoss) : undefined, secondsElapsed: elapsed() });
          await yieldToBrowser();
        }
      }
      const evaluation = await evaluateGraph(graph, prepared, data, options.batchSize);
      const record: EpochRecord = { epoch, loss: lossSum / countSum, accuracy: correctSum / countSum, evaluation, gradientNorms, secondsElapsed: elapsed() };
      epochs.push(record);
      onProgress({
        type: "epoch",
        task: kind,
        modelIndex,
        seed,
        run,
        epoch,
        batch: batches,
        batches,
        loss: record.loss,
        accuracy: record.accuracy,
        perplexity: tokenLevel ? Math.exp(record.loss) : undefined,
        valLoss: evaluation.loss,
        valAccuracy: evaluation.accuracy,
        valPerplexity: evaluation.perplexity,
        secondsElapsed: record.secondsElapsed,
        evaluation,
        gradientNorms,
      });
      await yieldToBrowser();
    }
    return result(false);
  } finally {
    optimizer.dispose();
    if (options.keepTrained) options.keepTrained({ modelIndex, run, seed, graph, prepared });
    else graph.dispose();
  }
}

/** One optimiser step. Returns the batch's mean loss, correct predictions and how many predictions were scored. */
function trainStep(graph: GraphModel, optimizer: tf.Optimizer, prepared: PreparedModel, data: TaskData, indices: Int32Array, step: number, clip: number, wantNorms: boolean) {
  let correct: tf.Scalar | undefined;
  let count: tf.Scalar | undefined;
  let norms: Record<string, number> | undefined;
  const loss = tf.tidy(() => {
    const batch = makeBatch(prepared.roles, data, "train", indices, graph);
    const { value, grads } = tf.variableGrads(() => {
      const output = graph.apply(batch.feeds, { training: true, step });
      const scored = scoreBatch(output, batch, graph.endsInSoftmax);
      correct = tf.keep(scored.correct);
      count = tf.keep(scored.count);
      return scored.loss;
    }, graph.trainable);
    const names = Object.keys(grads);
    const squares = names.map((name) => tf.sum(tf.square(grads[name])));
    if (wantNorms) {
      // Per-node gradient norms from this batch's gradients, summed over every copy of a repeated block.
      const values = tf.stack(squares).dataSync();
      const totals: Record<string, number> = {};
      names.forEach((name, i) => {
        const node = graph.nodeOfVariable.get(name);
        if (node) totals[node] = (totals[node] ?? 0) + values[i];
      });
      for (const node of Object.keys(totals)) totals[node] = Math.sqrt(totals[node]);
      norms = totals;
    }
    if (clip > 0 && names.length) {
      // Scale every gradient down together when their joint norm is above the limit.
      const norm = tf.sqrt(tf.addN(squares));
      const scale = tf.div(clip, tf.maximum(norm, clip));
      for (const name of names) grads[name] = tf.mul(grads[name], scale);
    }
    optimizer.applyGradients(grads);
    return value;
  });
  return { loss, correct: correct!, count: count!, norms };
}

/** Loss (mean over scored predictions), correct predictions and the count, from logits or probabilities. */
function scoreBatch(output: tf.Tensor, batch: Batch, probabilities: boolean) {
  const classes = output.shape[output.shape.length - 1];
  const flat = tf.reshape(output, [-1, classes]);
  const targets = tf.reshape(batch.targets, [-1]);
  const logProbs = probabilities ? tf.log(tf.clipByValue(flat, 1e-7, 1)) : tf.logSoftmax(flat);
  const nll = tf.neg(tf.sum(tf.mul(logProbs, tf.oneHot(targets, classes)), -1));
  const weights = batch.weights ? tf.reshape(batch.weights, [-1]) : tf.onesLike(nll);
  const count = tf.sum(weights) as tf.Scalar;
  const loss = tf.div(tf.sum(tf.mul(nll, weights)), tf.maximum(count, 1)) as tf.Scalar;
  const hits = tf.cast(tf.equal(tf.argMax(flat, -1), targets), "float32");
  const correct = tf.sum(tf.mul(hits, weights)) as tf.Scalar;
  return { loss, correct, count, nll, weights, flat, targets };
}

function makeBatch(roles: Record<string, InputRole>, data: TaskData, split: "train" | "test", indices: Int32Array, graph: GraphModel): Batch {
  const size = indices.length;
  if (data.image) {
    const index = tf.tensor1d(indices, "int32");
    const image = split === "train" ? data.image.xTrain : data.image.xTest;
    const labels = split === "train" ? data.image.yTrain : data.image.yTest;
    const feeds: Record<string, tf.Tensor> = {};
    for (const [id, role] of Object.entries(roles)) if (role === "image") feeds[id] = tf.gather(image, index);
    return { feeds, targets: tf.argMax(tf.gather(labels, index), -1) };
  }
  const part = (split === "train" ? data.train : data.test) as TokenSplit;
  const rows = (array: Int32Array | Float32Array, length: number) => {
    const out = array instanceof Int32Array ? new Int32Array(size * length) : new Float32Array(size * length);
    indices.forEach((example, i) => out.set(array.subarray(example * length, (example + 1) * length), i * length));
    return out;
  };
  const feeds: Record<string, tf.Tensor> = {};
  for (const [id, role] of Object.entries(roles)) {
    const length = graph.inputShapes[id][0];
    if (role === "zeros") feeds[id] = tf.zeros([size, length], "int32");
    else if (role === "tokens" || role === "source" || role === "target") feeds[id] = tf.tensor2d(rows(part.inputs[role]!, part.inputLength), [size, part.inputLength], "int32");
  }
  const T = part.targetLength;
  const targets = T === 1 ? tf.tensor1d(rows(part.targets, 1) as Int32Array, "int32") : tf.tensor2d(rows(part.targets, T), [size, T], "int32");
  const weights = part.weights ? tf.tensor2d(rows(part.weights, T), [size, T], "float32") : undefined;
  return { feeds, targets, weights };
}

/**
 * Test-split evaluation in the same shape as the image trainer's: loss,
 * accuracy, error rate, top-5, per-class scores and the confusion matrix,
 * counted over scored predictions (tokens for token tasks). Adds perplexity
 * for per-token tasks and whole-sequence accuracy for seq2seq.
 */
async function evaluateGraph(graph: GraphModel, prepared: PreparedModel, data: TaskData, batchSize: number): Promise<Evaluation> {
  const task = data.task;
  const n = data.image ? data.image.xTest.shape[0] : data.test!.size;
  const truth: number[] = [];
  const predicted: number[] = [];
  let lossSum = 0;
  let weightSum = 0;
  let top5Hits = 0;
  let sequencesRight = 0;
  for (let start = 0; start < n; start += batchSize) {
    const size = Math.min(batchSize, n - start);
    const indices = Int32Array.from({ length: size }, (_, i) => start + i);
    const [nllSum, wSum, argmax, labels, weights, top5] = tf.tidy(() => {
      const batch = makeBatch(prepared.roles, data, "test", indices, graph);
      const output = graph.apply(batch.feeds, { training: false });
      const s = scoreBatch(output, batch, graph.endsInSoftmax);
      const k = Math.min(5, task.classes);
      const inTop = tf.sum(tf.mul(tf.sum(tf.cast(tf.equal(tf.topk(s.flat, k).indices, tf.expandDims(s.targets, 1)), "float32"), 1), s.weights));
      return [tf.sum(tf.mul(s.nll, s.weights)), tf.sum(s.weights), tf.argMax(s.flat, -1), s.targets, s.weights, inTop];
    });
    const [l, w, a, t, ws, tops] = await Promise.all([nllSum.data(), wSum.data(), argmax.data(), labels.data(), weights.data(), top5.data()]);
    tf.dispose([nllSum, wSum, argmax, labels, weights, top5]);
    lossSum += l[0];
    weightSum += w[0];
    top5Hits += tops[0];
    const perExample = a.length / size;
    for (let e = 0; e < size; e++) {
      let allRight = true;
      for (let j = e * perExample; j < (e + 1) * perExample; j++) {
        if (ws[j] <= 0) continue;
        truth.push(t[j]);
        predicted.push(a[j]);
        if (a[j] !== t[j]) allRight = false;
      }
      if (allRight) sequencesRight++;
    }
  }
  const loss = lossSum / Math.max(1, weightSum);
  const evaluation = evaluate({ truth, predicted, top5Hits, loss, classes: task.classes });
  // Every scored token counts as one example here, so top-5 and accuracy are per token.
  if (task.tokenLevel) evaluation.perplexity = Math.exp(loss);
  if (task.kind === "seq2seq") evaluation.sequenceAccuracy = sequencesRight / n;
  return evaluation;
}

const yieldToBrowser = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function shuffled(n: number, random: () => number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
