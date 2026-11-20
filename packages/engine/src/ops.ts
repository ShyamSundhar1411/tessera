import type { ArgValue, Shape } from "./types";

// Every operation the engine knows: how it changes a tensor's shape, how many
// parameters it holds and how much compute it costs. Parameter counts follow
// PyTorch's `parameters()` (batch norm counts its scale and shift, not its
// running statistics), so totals match the numbers models are published with.

export type Category = "input" | "conv" | "linear" | "norm" | "activation" | "pool" | "merge" | "reshape" | "embedding" | "attention" | "mlp" | "output" | "regularise";

export interface OpResult {
  shape: Shape;
  params: number;
  activeParams?: number;
  flops: number;
}

type Args = Record<string, ArgValue>;

export interface OpSpec {
  category: Category;
  /** A one-line description for the inspector and the palette. */
  summary: string;
  /** How many inputs it takes: exactly one, or at least two. */
  arity: "one" | "many" | "none";
  /** The argument that a bare number in the text format sets, such as a convolution's output channels. */
  main?: string;
  infer(inputs: Shape[], args: Args): OpResult;
}

export class ShapeError extends Error {}

const num = (args: Args, key: string, fallback?: number): number => {
  const value = args[key];
  if (value === undefined || value === "") {
    if (fallback === undefined) throw new ShapeError(`needs ${key}`);
    return fallback;
  }
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) throw new ShapeError(`${key} must be a number`);
  return n;
};
const flag = (args: Args, key: string, fallback: boolean): boolean => {
  const value = args[key];
  if (value === undefined) return fallback;
  return value === true || value === "true" || value === 1 || value === "1";
};
const str = (args: Args, key: string, fallback: string): string => (args[key] === undefined ? fallback : String(args[key]));
const prod = (shape: Shape) => shape.reduce((a, b) => a * b, 1);
const same = (shape: Shape): OpResult => ({ shape: [...shape], params: 0, flops: prod(shape) });

const image = (shape: Shape, op: string): [number, number, number] => {
  if (shape.length !== 3) throw new ShapeError(`${op} needs an image-shaped input [channels, height, width], got [${shape.join(", ")}]`);
  return shape as [number, number, number];
};
const sequence = (shape: Shape, op: string): [number, number] => {
  if (shape.length !== 2) throw new ShapeError(`${op} needs a sequence-shaped input [seq, features], got [${shape.join(", ")}]`);
  return shape as [number, number];
};
const spatial = (size: number, k: number, s: number, p: number, d = 1) => Math.floor((size + 2 * p - d * (k - 1) - 1) / s) + 1;

/** Last-dimension size, the feature size for both sequences and flat vectors. */
const features = (shape: Shape) => shape[shape.length - 1];

const ACTIVATION = (summary: string): OpSpec => ({ category: "activation", summary, arity: "one", infer: ([x]) => same(x) });

export const OPS: Record<string, OpSpec> = {
  input: {
    category: "input",
    summary: "The data going in: an image [C, H, W], a token sequence [seq] or a vector [features].",
    arity: "none",
    infer: (_inputs, args) => {
      const shape = String(args.shape ?? "")
        .split(/[x×,\s]+/)
        .filter(Boolean)
        .map(Number);
      if (!shape.length || shape.some((n) => !Number.isFinite(n) || n <= 0)) throw new ShapeError("input needs a shape such as 3x224x224");
      return { shape, params: 0, flops: 0 };
    },
  },

  // -- Convolution and pooling ----------------------------------------------
  conv2d: {
    category: "conv",
    summary: "2D convolution: slides learned filters over the image.",
    arity: "one",
    main: "out",
    infer: ([x], args) => {
      const [c, h, w] = image(x, "conv2d");
      const out = num(args, "out");
      const k = num(args, "k", 3);
      const s = num(args, "s", 1);
      const p = num(args, "p", Math.floor(k / 2));
      const groups = args.groups === "depthwise" ? c : num(args, "groups", 1);
      if (c % groups || out % groups) throw new ShapeError(`groups ${groups} must divide both ${c} input and ${out} output channels`);
      const bias = flag(args, "bias", true);
      const weights = out * (c / groups) * k * k;
      const oh = spatial(h, k, s, p);
      const ow = spatial(w, k, s, p);
      if (oh <= 0 || ow <= 0) throw new ShapeError(`a ${k}×${k} filter with stride ${s} leaves nothing of a ${h}×${w} input`);
      return { shape: [out, oh, ow], params: weights + (bias ? out : 0), flops: 2 * weights * oh * ow };
    },
  },
  maxpool: {
    category: "pool",
    summary: "Keeps the largest value in each window, shrinking the image.",
    arity: "one",
    main: "k",
    infer: ([x], args) => {
      const [c, h, w] = image(x, "maxpool");
      const k = num(args, "k", 2);
      const s = num(args, "s", k);
      const p = num(args, "p", 0);
      const shape = [c, spatial(h, k, s, p), spatial(w, k, s, p)];
      return { shape, params: 0, flops: prod(shape) * k * k };
    },
  },
  avgpool: {
    category: "pool",
    summary: "Averages each window, shrinking the image.",
    arity: "one",
    main: "k",
    infer: ([x], args) => {
      const [c, h, w] = image(x, "avgpool");
      const k = num(args, "k", 2);
      const s = num(args, "s", k);
      const p = num(args, "p", 0);
      const shape = [c, spatial(h, k, s, p), spatial(w, k, s, p)];
      return { shape, params: 0, flops: prod(shape) * k * k };
    },
  },
  gap: {
    category: "pool",
    summary: "Global average pooling: one number per channel.",
    arity: "one",
    infer: ([x]) => {
      const [c, h, w] = image(x, "gap");
      return { shape: [c, 1, 1], params: 0, flops: c * h * w };
    },
  },

  // -- Dense layers ---------------------------------------------------------
  linear: {
    category: "linear",
    summary: "Fully connected layer: every input feature feeds every output.",
    arity: "one",
    main: "out",
    infer: ([x], args) => {
      const inF = features(x);
      const out = num(args, "out");
      const bias = flag(args, "bias", true);
      const rows = prod(x) / inF;
      return { shape: [...x.slice(0, -1), out], params: inF * out + (bias ? out : 0), flops: 2 * inF * out * rows };
    },
  },
  flatten: {
    category: "reshape",
    summary: "Lays the tensor out as one long vector.",
    arity: "one",
    infer: ([x]) => ({ shape: [prod(x)], params: 0, flops: 0 }),
  },
  identity: {
    category: "reshape",
    summary: "Passes its input on unchanged. Stands in for imported operations the engine does not model.",
    arity: "one",
    infer: ([x]) => ({ shape: [...x], params: 0, flops: 0 }),
  },

  // -- Normalisation and regularisation -------------------------------------
  batchnorm: {
    category: "norm",
    summary: "Batch normalisation: steadies each channel's values across a batch.",
    arity: "one",
    infer: ([x]) => ({ shape: [...x], params: 2 * x[0], flops: 2 * prod(x) }),
  },
  layernorm: {
    category: "norm",
    summary: "Layer normalisation over the features of each position.",
    arity: "one",
    infer: ([x], args) => ({ shape: [...x], params: (flag(args, "bias", true) ? 2 : 1) * features(x), flops: 5 * prod(x) }),
  },
  rmsnorm: {
    category: "norm",
    summary: "RMS normalisation: layer norm without the mean, used by most modern LLMs.",
    arity: "one",
    infer: ([x]) => ({ shape: [...x], params: features(x), flops: 4 * prod(x) }),
  },
  dropout: {
    category: "regularise",
    summary: "Randomly zeroes values while training, to reduce overfitting.",
    arity: "one",
    main: "rate",
    infer: ([x]) => ({ shape: [...x], params: 0, flops: 0 }),
  },

  // -- Activations ------------------------------------------------------------
  relu: ACTIVATION("ReLU: keeps positive values, zeroes the rest."),
  relu6: ACTIVATION("ReLU capped at 6, used by MobileNet."),
  gelu: ACTIVATION("GELU: a smooth ReLU, used by transformers."),
  silu: ACTIVATION("SiLU (swish): x times sigmoid(x), used by EfficientNet and Llama."),
  sigmoid: ACTIVATION("Sigmoid: squashes values into 0 to 1."),
  tanh: ACTIVATION("Tanh: squashes values into -1 to 1."),
  hardswish: ACTIVATION("Hard swish: a cheap approximation of SiLU."),
  softmax: ACTIVATION("Softmax: turns scores into probabilities."),

  // -- Joining paths ----------------------------------------------------------
  add: {
    category: "merge",
    summary: "Adds tensors of the same shape: the join of a skip (residual) connection.",
    arity: "many",
    infer: (inputs) => {
      const [first] = inputs;
      for (const shape of inputs.slice(1))
        if (shape.join() !== first.join()) throw new ShapeError(`add needs matching shapes, got [${first.join(", ")}] and [${shape.join(", ")}]`);
      return { shape: [...first], params: 0, flops: prod(first) * (inputs.length - 1) };
    },
  },
  mul: {
    category: "merge",
    summary: "Multiplies tensors, broadcasting [C, 1, 1] over [C, H, W]: a gate.",
    arity: "many",
    infer: (inputs) => {
      const out = inputs.reduce((a, b) => (prod(b) > prod(a) ? b : a));
      for (const shape of inputs) {
        if (shape.length !== out.length || shape.some((n, i) => n !== out[i] && n !== 1))
          throw new ShapeError(`mul needs shapes that broadcast, got [${shape.join(", ")}] and [${out.join(", ")}]`);
      }
      return { shape: [...out], params: 0, flops: prod(out) * (inputs.length - 1) };
    },
  },
  concat: {
    category: "merge",
    summary: "Joins tensors side by side along the channel or feature axis.",
    arity: "many",
    infer: (inputs, args) => {
      const rank = inputs[0].length;
      let axis = num(args, "axis", rank === 3 ? 0 : rank - 1);
      if (axis < 0) axis += rank;
      const out = [...inputs[0]];
      for (const shape of inputs.slice(1)) {
        if (shape.length !== rank || shape.some((n, i) => i !== axis && n !== out[i]))
          throw new ShapeError(`concat needs shapes that match except on axis ${axis}, got [${inputs[0].join(", ")}] and [${shape.join(", ")}]`);
        out[axis] += shape[axis];
      }
      return { shape: out, params: 0, flops: 0 };
    },
  },

  // -- Language models --------------------------------------------------------
  embedding: {
    category: "embedding",
    summary: "Looks up a learned vector for each token.",
    arity: "one",
    main: "dim",
    infer: ([x], args) => {
      if (x.length !== 1) throw new ShapeError(`embedding needs token ids [seq], got [${x.join(", ")}]`);
      const vocab = num(args, "vocab");
      const dim = num(args, "dim");
      return { shape: [x[0], dim], params: vocab * dim, flops: 0 };
    },
  },
  posembed: {
    category: "embedding",
    summary: "Learned position embeddings added to each token (GPT-2, BERT).",
    arity: "one",
    infer: ([x], args) => {
      const [seq, dim] = sequence(x, "posembed");
      const max = num(args, "max", seq);
      return { shape: [seq, dim], params: max * dim, flops: seq * dim };
    },
  },
  attention: {
    category: "attention",
    summary: "Self-attention. Heads share key/value heads when kv < heads (grouped-query attention).",
    arity: "one",
    main: "heads",
    infer: ([x], args) => {
      const [seq, d] = sequence(x, "attention");
      const heads = num(args, "heads");
      const kv = num(args, "kv", heads);
      const hd = num(args, "head_dim", Math.floor(d / heads));
      if (heads % kv) throw new ShapeError(`${heads} heads cannot be shared evenly by ${kv} key/value heads`);
      const qkvBias = flag(args, "qkv_bias", false);
      const oBias = flag(args, "o_bias", false);
      const qkNorm = flag(args, "qk_norm", false);
      const weights = d * heads * hd + 2 * d * kv * hd + heads * hd * d;
      const params = weights + (qkvBias ? heads * hd + 2 * kv * hd : 0) + (oBias ? d : 0) + (qkNorm ? 2 * hd : 0);
      const window = num(args, "window", 0);
      const span = window > 0 ? Math.min(window, seq) : seq;
      const scores = 2 * 2 * seq * span * heads * hd;
      return { shape: [seq, d], params, flops: 2 * weights * seq + scores };
    },
  },
  mla: {
    category: "attention",
    summary: "Multi-head latent attention (DeepSeek): keys and values squeezed through a small shared latent.",
    arity: "one",
    main: "heads",
    infer: ([x], args) => {
      const [seq, d] = sequence(x, "mla");
      const heads = num(args, "heads");
      const qRank = num(args, "q_rank", 0);
      const kvRank = num(args, "kv_rank");
      const nope = num(args, "qk_nope_dim");
      const rope = num(args, "qk_rope_dim");
      const v = num(args, "v_dim");
      const q = qRank > 0 ? d * qRank + qRank + qRank * heads * (nope + rope) : d * heads * (nope + rope);
      const kvDown = d * (kvRank + rope) + kvRank;
      const kvUp = kvRank * heads * (nope + v);
      const out = heads * v * d;
      const params = q + kvDown + kvUp + out;
      const scores = 2 * seq * seq * heads * (nope + rope) + 2 * seq * seq * heads * v;
      return { shape: [seq, d], params, flops: 2 * params * seq + scores };
    },
  },
  mlp: {
    category: "mlp",
    summary: "The feed-forward network of a transformer layer. Gated (SwiGLU) uses three matrices, plain uses two.",
    arity: "one",
    main: "ff",
    infer: ([x], args) => {
      const d = features(x);
      const ff = num(args, "ff");
      const gated = str(args, "act", "swiglu") !== "gelu" && str(args, "act", "swiglu") !== "relu";
      const bias = flag(args, "bias", false);
      const params = (gated ? 3 : 2) * d * ff + (bias ? ff + d : 0);
      return { shape: [...x], params, flops: 2 * params * (prod(x) / d) };
    },
  },
  moe: {
    category: "mlp",
    summary: "Mixture of experts: a router sends each token to a few of many feed-forward experts.",
    arity: "one",
    main: "experts",
    infer: ([x], args) => {
      const d = features(x);
      const experts = num(args, "experts");
      const topk = num(args, "topk", 2);
      const ff = num(args, "ff");
      const shared = num(args, "shared", 0);
      const sharedFf = num(args, "shared_ff", ff);
      const sharedGate = flag(args, "shared_gate", false);
      const routerBias = flag(args, "router_bias", false);
      if (topk > experts) throw new ShapeError(`top-${topk} needs at least ${topk} experts`);
      const expert = 3 * d * ff;
      const sharedParams = shared * 3 * d * sharedFf + (sharedGate ? d : 0);
      const router = d * experts + (routerBias ? experts : 0);
      const params = router + experts * expert + sharedParams;
      const activeParams = router + topk * expert + sharedParams;
      return { shape: [...x], params, activeParams, flops: 2 * activeParams * (prod(x) / d) };
    },
  },
  lmhead: {
    category: "output",
    summary: "Turns each position into scores over the vocabulary. Often shares its weights with the embedding.",
    arity: "one",
    main: "vocab",
    infer: ([x], args) => {
      const [seq, d] = sequence(x, "lmhead");
      const vocab = num(args, "vocab");
      const tied = flag(args, "tied", false);
      return { shape: [seq, vocab], params: tied ? 0 : d * vocab, flops: 2 * d * vocab * seq };
    },
  },
};

export const OP_NAMES = Object.keys(OPS);

/** Short names accepted in the text format. */
export const ALIASES: Record<string, string> = {
  conv: "conv2d",
  bn: "batchnorm",
  ln: "layernorm",
  norm: "rmsnorm",
  fc: "linear",
  dense: "linear",
  pool: "maxpool",
  attn: "attention",
  swish: "silu",
  embed: "embedding",
  head: "lmhead",
  cat: "concat",
};
