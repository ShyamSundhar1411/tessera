import { analyze, shrinkLanguageModel, type ArgValue, type Model, type Node } from "tessera-engine";
import { DATASETS, type Dataset, type DatasetName } from "./datasets";
import { mulberry32 } from "./train";

// What a model should learn in the browser, decided from its wiring, and the
// small version of it that learns it in seconds to minutes.
//
// - A decoder (ends in lmhead, causal attention): predict the next character
//   of tiny Shakespeare.
// - An encoder (bidirectional attention) that ends per token: fill in masked
//   characters, BERT style. One that ends in a single vector: classify a
//   generated sequence.
// - An encoder-decoder (cross_attention, two inputs): turn a random sequence
//   into its reverse (or its sorted order), with teacher forcing.
// - Anything that reads images, CNN or vision transformer: MNIST or CIFAR-10.
//
// Transformers are shrunk with the engine's shrinkLanguageModel, keeping their
// wiring (attention kind, head sharing, gating, experts, norms) and only
// changing sizes, then their inputs and heads are fitted to the task.

export type TaskKind = "language-modelling" | "masked-tokens" | "seq2seq" | "sequence-classification" | "image-classification";

export interface TaskSpec {
  kind: TaskKind;
  /** One line for the UI, such as "Next-character prediction on tiny Shakespeare". */
  label: string;
  /** What "accuracy" counts for this task. */
  accuracyLabel: string;
  /** Where the data comes from: a URL, or "generated in the browser". */
  source: string;
  /** How many classes each prediction chooses between: the vocabulary for token tasks. */
  classes: number;
  /** Token ids the model reads (equals classes for character tasks). */
  vocab?: number;
  /** Sequence length of the token inputs. */
  seqLen?: number;
  /** True for tasks scored per token, where perplexity applies. */
  tokenLevel: boolean;
  /** For image tasks: which dataset to load with loadDataset. */
  imageDataset?: DatasetName;
  /** For seq2seq: what the decoder must produce from the source. */
  seq2seq?: "reverse" | "sort";
  /** Two models can be compared only when this matches. */
  key: string;
}

/** What each input node of a prepared model is fed. "zeros" is for extra token inputs such as segment ids. */
export type InputRole = "tokens" | "source" | "target" | "image" | "zeros";

export interface PreparedModel {
  ok: true;
  task: TaskSpec;
  /** The shrunk or adapted model that will train. */
  model: Model;
  /** Plain words on what was changed and what it learns. */
  description: string;
  params: number;
  originalParams: number;
  /** Input node id to what it is fed. */
  roles: Record<string, InputRole>;
}

export type PrepareResult = PreparedModel | { ok: false; reason: string };

export interface PrepareOptions {
  /** Hidden size of shrunk transformers (default 64). */
  dim?: number;
  /** Total transformer layers after shrinking, shared across the model's stacks (default 4). */
  layers?: number;
  /** Attention heads after shrinking (default 4). */
  heads?: number;
  /** Characters of context for language tasks (default 64). */
  seqLen?: number;
  /** Source and target length for the seq2seq task (default 12). */
  seq2seqLength?: number;
  seq2seqTask?: "reverse" | "sort";
  /** Sequence length for the generated classification task (default 32). */
  classifyLength?: number;
  /** For an encoder that ends in a pooled vector: fill in masked characters under the pooler (default), or classify a generated sequence. */
  encoderTask?: "masked-tokens" | "sequence-classification";
  /** Override the image dataset picked from the input channels (mnist for 1, cifar10 for 3). */
  imageDataset?: DatasetName;
}

// -- Vocabularies -------------------------------------------------------------------

/** Characters: newline, printable ASCII 32..126, then one id for anything else. Fixed, so a model can be sized before the text arrives. */
export const CHAR_VOCAB = 97;
export const CHAR_UNKNOWN = 96;
/** The masked-token task adds one id for [MASK]. */
export const CHAR_MASK = 97;

export function encodeChars(text: string): Int32Array {
  const out = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    out[i] = c === 10 ? 0 : c >= 32 && c <= 126 ? c - 31 : CHAR_UNKNOWN;
  }
  return out;
}

export function decodeChars(ids: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    s += id === 0 ? "\n" : id >= 1 && id <= 95 ? String.fromCharCode(id + 31) : id === CHAR_MASK ? "_" : "?";
  }
  return s;
}

/** Sequence tasks draw from 16 symbols, 0-9 and a-f. seq2seq puts a start token at id 0, so its symbols are 1..16. */
export const SYMBOLS = "0123456789abcdef";
const CLASSIFY_CLASSES = 4;

export const SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

// -- Reading the wiring -------------------------------------------------------------

const isTrue = (value: ArgValue | undefined) => value === true || value === "true" || value === 1 || value === "1";
const SEQUENCE_OPS = new Set(["attention", "cross_attention", "mla", "to_sequence", "cls_token", "embedding", "posembed", "lmhead", "moe", "mlp"]);

/** Node ids `start` depends on, following inputs back. */
function ancestors(model: Model, start: string): Set<string> {
  const byId = new Map(model.nodes.map((node) => [node.id, node]));
  const seen = new Set<string>();
  const stack = [start];
  while (stack.length) {
    const id = stack.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const input of byId.get(id)?.inputs ?? []) stack.push(input);
  }
  return seen;
}

const count = (n: number) => n.toLocaleString("en-US");

/**
 * Picks the task for a model from its wiring and returns the version of it
 * that trains in the browser, or a reason it cannot.
 */
export function prepareForTraining(model: Model, options: PrepareOptions = {}): PrepareResult {
  const analysis = analyze(model);
  if (analysis.errors.length) return { ok: false, reason: `${model.name} has a problem at "${analysis.errors[0].node}": ${analysis.errors[0].message}` };
  const inputs = model.nodes.filter((node) => node.op === "input");
  if (!inputs.length) return { ok: false, reason: `${model.name} has no input` };
  const shapeOf = (id: string) => analysis.info.get(id)!.shape;
  const originalParams = analysis.params;
  const readers = (id: string) => model.nodes.filter((node) => node.inputs.includes(id));

  if (inputs.some((node) => shapeOf(node.id).length === 3)) return prepareImage(model, options, originalParams);

  const tokenInputs = inputs.filter((node) => shapeOf(node.id).length === 1 && readers(node.id).some((r) => r.op === "embedding"));
  if (!tokenInputs.length)
    return { ok: false, reason: `${model.name} reads [${shapeOf(inputs[0].id).join(", ")}] vectors, not images or tokens; the browser trainer has no dataset for that shape yet` };
  if (model.kind !== "language")
    return { ok: false, reason: `${model.name} reads tokens but is not marked as a language model, so it cannot be shrunk for the browser` };

  const crosses = model.nodes.filter((node) => node.op === "cross_attention");
  if (crosses.length) return prepareSeq2seq(model, options, originalParams, crosses);

  const attentions = model.nodes.filter((node) => node.op === "attention" || node.op === "mla");
  const bidirectional = attentions.some((node) => node.args.causal !== undefined && !isTrue(node.args.causal));
  const out = analysis.outputShape ?? [];
  const last = model.nodes[model.nodes.length - 1];
  let working = model;
  let kind: TaskKind;
  const changes: string[] = [];

  if (last.op === "lmhead") kind = bidirectional ? "masked-tokens" : "language-modelling";
  else if (out.length === 2) {
    // Ends per token without an output layer (DistilBERT, a bare decoder): add one, tied to the embedding as BERT's is.
    kind = bidirectional ? "masked-tokens" : "language-modelling";
    working = appendNode(model, "lmhead", last.id, { vocab: 1, tied: true }, "Output (added for training)");
    changes.push("added an output layer tied to the embedding");
  } else if (out.length === 1) {
    const pool = model.nodes.find((node) => node.op === "take_first" || node.op === "mean_pool");
    const smallHead = last.op === "linear" && Number(last.args.out) <= 64;
    // An encoder with a pooler (BERT, RoBERTa) trains the way BERT was
    // trained, so it lines up with encoders that end per token (DistilBERT).
    // A model that already ends in a small classifier keeps it and classifies.
    const masked = pool && bidirectional && (options.encoderTask === "masked-tokens" || (!smallHead && options.encoderTask !== "sequence-classification"));
    if (masked) {
      kind = "masked-tokens";
      const keep = model.nodes.slice(0, model.nodes.indexOf(pool));
      working = appendNode({ ...model, nodes: keep }, "lmhead", pool.inputs[0], { vocab: 1, tied: true }, "Output (added for training)");
      changes.push("swapped the pooler for an output layer tied to the embedding");
    } else {
      kind = "sequence-classification";
      if (!smallHead) {
        working = appendNode(model, "linear", last.id, { out: CLASSIFY_CLASSES }, "Classifier (added for training)");
        changes.push(`added a ${CLASSIFY_CLASSES}-way classifier after the pooled output`);
      }
    }
  } else return { ok: false, reason: `${model.name} ends in [${out.join(", ")}], which is neither per-token scores nor one vector` };

  const seqLen = kind === "sequence-classification" ? (options.classifyLength ?? 32) : (options.seqLen ?? 64);
  const vocab = kind === "masked-tokens" ? CHAR_VOCAB + 1 : kind === "sequence-classification" ? SYMBOLS.length : CHAR_VOCAB;
  let small = shrink(working, options, vocab, seqLen);
  for (const node of small.nodes) if (node.op === "input") node.args = { ...node.args, shape: String(seqLen) };
  if (kind === "sequence-classification") small = setLastLinear(small, CLASSIFY_CLASSES);

  const task: TaskSpec =
    kind === "language-modelling"
      ? { kind, label: "Next-character prediction on tiny Shakespeare", accuracyLabel: "next-character accuracy", source: SHAKESPEARE_URL, classes: CHAR_VOCAB, vocab: CHAR_VOCAB, seqLen, tokenLevel: true, key: `lm:${seqLen}` }
      : kind === "masked-tokens"
        ? { kind, label: "Masked-character prediction on tiny Shakespeare (BERT style, 15% hidden)", accuracyLabel: "masked-character accuracy", source: SHAKESPEARE_URL, classes: CHAR_VOCAB + 1, vocab: CHAR_VOCAB + 1, seqLen, tokenLevel: true, key: `mlm:${seqLen}` }
        : {
            kind,
            label: `Sequence classification: which of 0, 1, 2, 3 appears most in ${seqLen} random symbols`,
            accuracyLabel: "classification accuracy",
            source: "generated in the browser",
            classes: CLASSIFY_CLASSES,
            vocab: SYMBOLS.length,
            seqLen,
            tokenLevel: false,
            key: `cls:${seqLen}`,
          };
  const roles: Record<string, InputRole> = {};
  // The first token input carries the text; any other (segment ids, say) is fed zeros.
  for (const node of inputs) roles[node.id] = node.id === tokenInputs[0].id ? "tokens" : "zeros";
  return finish(model, small, task, roles, originalParams, changes);
}

function prepareSeq2seq(model: Model, options: PrepareOptions, originalParams: number, crosses: Node[]): PrepareResult {
  const inputs = model.nodes.filter((node) => node.op === "input");
  // The source is whatever feeds the encoder output that cross-attention reads; the target feeds the decoder stream.
  const memorySide = new Set(crosses.flatMap((node) => [...ancestors(model, node.inputs[1])]));
  const decoderSide = new Set(crosses.flatMap((node) => [...ancestors(model, node.inputs[0])]));
  const source = inputs.find((node) => memorySide.has(node.id));
  const target = inputs.find((node) => decoderSide.has(node.id) && !memorySide.has(node.id));
  if (!source || !target) return { ok: false, reason: `${model.name} has cross-attention but its source and target inputs could not be told apart` };
  const last = model.nodes[model.nodes.length - 1];
  if (last.op !== "lmhead") return { ok: false, reason: `${model.name} is an encoder-decoder that does not end in an output layer (lmhead), so it cannot predict target tokens` };
  const length = options.seq2seqLength ?? 12;
  const which = options.seq2seqTask ?? "reverse";
  const vocab = SYMBOLS.length + 1;
  const small = shrink(model, options, vocab, length);
  for (const node of small.nodes) if (node.op === "input") node.args = { ...node.args, shape: String(length) };
  const task: TaskSpec = {
    kind: "seq2seq",
    label: `${which === "reverse" ? "Reverse" : "Sort"} a random sequence of ${length} symbols (encoder reads it, decoder writes the answer)`,
    accuracyLabel: "target-token accuracy",
    source: "generated in the browser",
    classes: vocab,
    vocab,
    seqLen: length,
    tokenLevel: true,
    seq2seq: which,
    key: `seq2seq:${which}:${length}`,
  };
  const roles: Record<string, InputRole> = {};
  for (const node of inputs) roles[node.id] = node.id === source.id ? "source" : node.id === target.id ? "target" : "zeros";
  return finish(model, small, task, roles, originalParams, []);
}

function prepareImage(model: Model, options: PrepareOptions, originalParams: number): PrepareResult {
  const inputs = model.nodes.filter((node) => node.op === "input");
  if (inputs.length !== 1) return { ok: false, reason: `${model.name} has ${inputs.length} inputs; image training needs exactly one` };
  const input = inputs[0];
  const channels = Number(String(input.args.shape).split(/[x×,\s]+/)[0]);
  const name: DatasetName | undefined = options.imageDataset ?? (channels === 1 ? "mnist" : channels === 3 ? "cifar10" : undefined);
  if (!name) return { ok: false, reason: `${model.name} reads ${channels}-channel images; the browser datasets have 1 (MNIST) or 3 (CIFAR-10)` };
  const info = DATASETS[name];
  if (info.shape[0] !== channels) return { ok: false, reason: `${model.name} reads ${channels}-channel images but ${info.label} has ${info.shape[0]}` };
  const [, h, w] = info.shape;
  const transformer = model.nodes.some((node) => node.op === "to_sequence");
  const changes: string[] = [];
  let small: Model;
  if (transformer) {
    // A vision transformer: shrink its layers, then cut the small images into 4×4 patches.
    small = shrink(model, options, undefined, 64);
    const patches = small.nodes.find((node) => node.op === "conv2d" && small.nodes.some((other) => other.op === "to_sequence" && other.inputs.includes(node.id)));
    if (patches) patches.args = { ...patches.args, k: 4, s: 4, p: 0 };
    changes.push(`patches of 4×4 pixels (${(h / 4) * (w / 4)} tokens)`);
  } else small = { ...model, nodes: model.nodes.map((node) => ({ ...node, args: { ...node.args } })), groups: model.groups.map((g) => ({ ...g })) };
  const smallInput = small.nodes.find((node) => node.op === "input")!;
  if (String(smallInput.args.shape) !== info.shape.join("x")) {
    smallInput.args = { ...smallInput.args, shape: info.shape.join("x") };
    changes.push(`input set to ${info.shape.join("×")}`);
  }
  small = setLastLinear(small, info.classes.length, changes);
  const task: TaskSpec = {
    kind: "image-classification",
    label: `Image classification on ${info.label}`,
    accuracyLabel: "classification accuracy",
    source: name === "mnist" ? "MNIST (Google Cloud Storage tf.js sprite)" : name === "cifar10" ? "CIFAR-10 (Hugging Face uoft-cs/cifar10)" : `${info.label} (GitHub)`,
    classes: info.classes.length,
    tokenLevel: false,
    imageDataset: name,
    key: `image:${name}`,
  };
  return finish(model, small, task, { [input.id]: "image" }, originalParams, changes, !transformer);
}

/** Shrinks with the engine, then puts back what shrinking should not touch for training. */
function shrink(model: Model, options: PrepareOptions, vocab: number | undefined, seqLen: number): Model {
  const small = shrinkLanguageModel(model, { dim: options.dim ?? 64, layers: options.layers ?? 4, heads: options.heads ?? 4, vocab, seqLen, maxExperts: 4 });
  // shrinkLanguageModel shares its layer budget over every top-level group,
  // including ones that never repeated (a pooler, a classifier); those must
  // stay single or their shapes stop matching.
  const groups = small.groups.map((group) => {
    const original = model.groups.find((g) => g.id === group.id);
    return (original?.repeat ?? 1) <= 1 ? { ...group, repeat: original?.repeat } : group;
  });
  const nodes = small.nodes.map((node) => {
    const args = { ...node.args };
    // A learned position table needs only as many rows as the training sequence; leaving max out makes it exactly that long.
    if (node.op === "posembed" && !isTrue(args.types) && !isTrue(args.fixed)) delete args.max;
    if ((node.op === "embedding" || node.op === "lmhead") && vocab) args.vocab = vocab;
    return { ...node, args };
  });
  return { ...small, nodes, groups };
}

/** Points the model's final linear layer (before a closing softmax, if any) at the task's classes. */
function setLastLinear(model: Model, classes: number, changes?: string[]): Model {
  const nodes = model.nodes.map((node) => ({ ...node, args: { ...node.args } }));
  let i = nodes.length - 1;
  if (nodes[i].op === "softmax") i--;
  if (nodes[i]?.op === "linear" && Number(nodes[i].args.out) !== classes) {
    nodes[i].args.out = classes;
    changes?.push(`last layer set to ${classes} classes`);
  }
  return { ...model, nodes };
}

function appendNode(model: Model, op: string, input: string, args: Record<string, ArgValue>, label: string): Model {
  let id = `task_${op}`;
  while (model.nodes.some((node) => node.id === id)) id += "_";
  return { ...model, nodes: [...model.nodes, { id, op, args, inputs: [input], label }] };
}

function finish(original: Model, small: Model, task: TaskSpec, roles: Record<string, InputRole>, originalParams: number, changes: string[], unshrunk = false): PrepareResult {
  const analysis = analyze(small);
  if (analysis.errors.length) return { ok: false, reason: `${original.name} could not be fitted to "${task.label}": ${analysis.errors[0].message} (at "${analysis.errors[0].node}")` };
  const out = analysis.outputShape ?? [];
  const wantsTokens = task.tokenLevel;
  if (wantsTokens ? out[out.length - 1] !== task.classes : out.length !== 1 || out[0] !== task.classes)
    return { ok: false, reason: `${original.name} ends in [${out.join(", ")}] but "${task.label}" needs ${wantsTokens ? `scores over ${task.classes} tokens per position` : `${task.classes} class scores`}` };
  const size = unshrunk ? `${count(analysis.params)} parameters` : `shrunk to ${small.meta.shrunk ?? "a small size"}: ${count(analysis.params)} parameters, from ${count(originalParams)}`;
  const description = `${original.name}, ${size}${changes.length ? `; ${changes.join(", ")}` : ""}. Task: ${task.label}.`;
  return { ok: true, task, model: { ...small, name: unshrunk ? original.name : small.name }, description, params: analysis.params, originalParams, roles };
}

/** Prepares every model and checks they share one task. Returns the shared task or a message saying why they cannot be compared. */
export function planTraining(models: Model[], options: PrepareOptions = {}): { ok: true; task: TaskSpec; prepared: PreparedModel[] } | { ok: false; reason: string } {
  if (!models.length) return { ok: false, reason: "no models to train" };
  const prepared: PreparedModel[] = [];
  for (const model of models) {
    const result = prepareForTraining(model, options);
    if (!result.ok) return result;
    prepared.push(result);
  }
  const first = prepared[0];
  const other = prepared.find((p) => p.task.key !== first.task.key);
  if (other) {
    const a = models[0].name;
    const b = models[prepared.indexOf(other)].name;
    return { ok: false, reason: `${a} and ${b} learn different things, so their curves cannot be compared: ${a} would do "${first.task.label}", ${b} would do "${other.task.label}". Pick two models of the same kind (two decoders, two encoders, two encoder-decoders, or two image models with the same input channels).` };
  }
  return { ok: true, task: first.task, prepared };
}

// -- Data ------------------------------------------------------------------------

/** One split of a token task, as flat arrays: example i occupies [i*len, (i+1)*len). */
export interface TokenSplit {
  size: number;
  /** Input arrays by role; each example has `inputLength` ids. */
  inputs: Partial<Record<"tokens" | "source" | "target", Int32Array>>;
  inputLength: number;
  /** Targets: `targetLength` per example (1 for classification). */
  targets: Int32Array;
  targetLength: number;
  /** Which target positions count (masked-token task); all count when absent. */
  weights?: Float32Array;
}

export interface TaskData {
  task: TaskSpec;
  train: TokenSplit | null;
  test: TokenSplit | null;
  /** Image tasks keep the loaded dataset's tensors. */
  image?: Dataset;
}

export interface TaskDataOptions {
  /** Text for the character tasks; tiny Shakespeare is fetched when absent. */
  text?: string;
  /** Training examples (windows or generated sequences). Default 2048. */
  trainExamples?: number;
  /** Test examples. Default 256. */
  testExamples?: number;
  /** Seeds the data itself (window positions, masks, generated sequences); the same for every model and run. Default 1234. */
  dataSeed?: number;
  /** The loaded dataset, for image tasks. */
  image?: Dataset;
}

let shakespeare: Promise<string> | undefined;
/** Tiny Shakespeare (1.1 MB, served with Access-Control-Allow-Origin: *), fetched once per page. */
export function loadShakespeare(): Promise<string> {
  shakespeare ??= fetch(SHAKESPEARE_URL).then((response) => {
    if (!response.ok) throw new Error(`could not download tiny Shakespeare (${response.status})`);
    return response.text();
  });
  shakespeare.catch(() => (shakespeare = undefined));
  return shakespeare;
}

export async function loadTaskData(task: TaskSpec, options: TaskDataOptions = {}): Promise<TaskData> {
  const train = options.trainExamples ?? 2048;
  const test = options.testExamples ?? 256;
  const seed = options.dataSeed ?? 1234;
  if (task.kind === "image-classification") {
    if (!options.image) throw new Error(`"${task.label}" needs the image dataset: load ${task.imageDataset} with loadDataset and pass it as imageDataset`);
    return { task, train: null, test: null, image: options.image };
  }
  if (task.kind === "language-modelling" || task.kind === "masked-tokens") {
    const text = options.text ?? (await loadShakespeare());
    return makeCharData(task, encodeChars(text), train, test, seed);
  }
  return makeGeneratedData(task, train, test, seed);
}

/** Windows of text. The first 90% of the text gives training windows, the rest test windows, so the test is text never trained on. */
export function makeCharData(task: TaskSpec, ids: Int32Array, trainCount: number, testCount: number, seed: number): TaskData {
  const L = task.seqLen!;
  const lm = task.kind === "language-modelling";
  const span = lm ? L + 1 : L;
  if (ids.length < span + 1) throw new Error(`the text has ${ids.length} characters; it needs at least ${span + 1}`);
  const cut = Math.floor(ids.length * 0.9);
  // Short texts (tests, pasted snippets) cannot spare a held-out part, so both splits read the whole text.
  const [trainStart, trainEnd, testStart, testEnd] = cut - span < span || ids.length - cut < span ? [0, ids.length, 0, ids.length] : [0, cut, cut, ids.length];
  const random = mulberry32(seed);
  const trainOffsets = Array.from({ length: trainCount }, () => trainStart + Math.floor(random() * (trainEnd - trainStart - span + 1)));
  const room = testEnd - testStart - span;
  const testOffsets = Array.from({ length: testCount }, (_, i) => testStart + Math.floor((i * room) / Math.max(1, testCount)));
  const build = (offsets: number[], rng: () => number): TokenSplit => {
    const n = offsets.length;
    const inputs = new Int32Array(n * L);
    const targets = new Int32Array(n * L);
    const weights = lm ? undefined : new Float32Array(n * L);
    offsets.forEach((offset, e) => {
      for (let t = 0; t < L; t++) {
        if (lm) {
          inputs[e * L + t] = ids[offset + t];
          targets[e * L + t] = ids[offset + t + 1];
        } else {
          // BERT's recipe: hide 15% of positions; of those, 80% become [MASK], 10% a random character, 10% stay.
          const id = ids[offset + t];
          targets[e * L + t] = id;
          inputs[e * L + t] = id;
          if (rng() < 0.15) {
            weights![e * L + t] = 1;
            const r = rng();
            inputs[e * L + t] = r < 0.8 ? CHAR_MASK : r < 0.9 ? 1 + Math.floor(rng() * 95) : id;
          }
        }
      }
      // Every example scores at least one position.
      if (weights && !weights.subarray(e * L, (e + 1) * L).some((v) => v > 0)) {
        weights[e * L] = 1;
        inputs[e * L] = CHAR_MASK;
      }
    });
    return { size: n, inputs: { tokens: inputs }, inputLength: L, targets, targetLength: L, weights };
  };
  return { task, train: build(trainOffsets, mulberry32(seed + 1)), test: build(testOffsets, mulberry32(seed + 2)) };
}

/** Generated sequences for the seq2seq and classification tasks. */
export function makeGeneratedData(task: TaskSpec, trainCount: number, testCount: number, seed: number): TaskData {
  const L = task.seqLen!;
  const k = SYMBOLS.length;
  const build = (n: number, random: () => number): TokenSplit => {
    if (task.kind === "seq2seq") {
      // Symbols are ids 1..16; id 0 starts the decoder. The decoder reads the
      // answer shifted right by one (teacher forcing) and predicts it whole.
      const source = new Int32Array(n * L);
      const target = new Int32Array(n * L);
      const targets = new Int32Array(n * L);
      for (let e = 0; e < n; e++) {
        const s = Array.from({ length: L }, () => 1 + Math.floor(random() * k));
        const answer = task.seq2seq === "sort" ? [...s].sort((a, b) => a - b) : [...s].reverse();
        for (let t = 0; t < L; t++) {
          source[e * L + t] = s[t];
          targets[e * L + t] = answer[t];
          target[e * L + t] = t === 0 ? 0 : answer[t - 1];
        }
      }
      return { size: n, inputs: { source, target }, inputLength: L, targets, targetLength: L };
    }
    // Classification: which of symbols 0..3 occurs most (ties go to the lower one).
    const tokens = new Int32Array(n * L);
    const targets = new Int32Array(n);
    for (let e = 0; e < n; e++) {
      const counts = new Array(CLASSIFY_CLASSES).fill(0);
      for (let t = 0; t < L; t++) {
        const id = Math.floor(random() * k);
        tokens[e * L + t] = id;
        if (id < CLASSIFY_CLASSES) counts[id]++;
      }
      targets[e] = counts.indexOf(Math.max(...counts));
    }
    return { size: n, inputs: { tokens }, inputLength: L, targets, targetLength: 1 };
  };
  return { task, train: build(trainCount, mulberry32(seed)), test: build(testCount, mulberry32(seed + 1)) };
}
