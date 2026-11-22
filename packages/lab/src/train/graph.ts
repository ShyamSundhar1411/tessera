import * as tf from "@tensorflow/tfjs";
import { analyze, type Analysis, type ArgValue, type Model, type Node, type Shape } from "tessera-engine";
import { hashSeed, isWithin, outermostRepeated } from "./toTf";

// Runs ANY engine graph in tf.js: convolutional networks, decoder-only
// language models, encoders, encoder-decoders and vision transformers.
//
// toTf.ts builds tf.layers models, which only cover image layers. Here every
// op is written with plain tf ops over tf.variables instead, so attention,
// mixture of experts, tied tables and the rest are a few lines each. The maths
// follows the engine's PyTorch export (packages/engine/src/export/pytorch.ts),
// and every node holds exactly the parameters the engine counts for it, so
// the number shown in the app is the number being trained.
//
// Layouts inside tf: images are channels-last [batch, H, W, C] (the engine is
// channels-first [C, H, W]); sequences are [batch, seq, features]; token ids
// are int32 [batch, seq].

export class GraphError extends Error {}

export interface GraphOptions {
  /** Seeds the initial weights. Each tensor's seed comes from this and its node id, as in toTf, so two variants of a model start alike where they match. */
  seed?: number;
}

export interface ApplyOptions {
  /** Training mode: dropout is on and batch norm uses batch statistics (and updates its running averages). */
  training: boolean;
  /** Mixed into dropout seeds so every step draws a fresh, reproducible mask. */
  step?: number;
}

export interface GraphModel {
  model: Model;
  analysis: Analysis;
  /** Input nodes in graph order. Feed them by id. */
  inputs: Node[];
  /** The engine's shape of every input (no batch). */
  inputShapes: Record<string, Shape>;
  /** The node the output comes from (the last one) and its engine shape. */
  outputNode: Node;
  outputShape: Shape;
  /** True when the graph ends in its own softmax, so the output is probabilities rather than logits. */
  endsInSoftmax: boolean;
  /** Everything the optimiser updates. */
  trainable: tf.Variable[];
  /** Which engine node each trainable variable belongs to (by variable name). Copies of a repeated block map to the one node. */
  nodeOfVariable: Map<string, string>;
  /** Total trainable values; equals analyze(model).params. */
  paramCount: number;
  /** Runs the graph. Call inside tf.tidy (or a gradient function); the result is the last node's output. */
  apply(feeds: Record<string, tf.Tensor>, options: ApplyOptions): tf.Tensor;
  /**
   * Runs the graph in evaluation mode and keeps what every layer produced, for
   * explaining a forward pass: each step's output, and for attention layers
   * the attention weights [batch, heads, query, key]. The caller disposes the
   * tensors (disposeTrace).
   */
  trace(feeds: Record<string, tf.Tensor>): TraceStep[];
  dispose(): void;
}

export interface TraceStep {
  /** The unrolled copy, such as "attention_r3" for the fourth copy of a repeated block. */
  key: string;
  nodeId: string;
  /** Which copy of a repeated block this is (0 when not repeated). */
  copy: number;
  value: tf.Tensor;
  attention?: tf.Tensor;
}

export function disposeTrace(steps: TraceStep[]): void {
  for (const step of steps) {
    step.value.dispose();
    step.attention?.dispose();
  }
}

/** Set while tracing, so attention can hand back its weights without changing every signature. */
let attentionSink: tf.Tensor[] | null = null;

interface Step {
  /** Unique per unrolled copy: node id plus a copy suffix such as "_r0". */
  key: string;
  node: Node;
  /** Keys of the steps this one reads. */
  inputs: string[];
  suffix: string;
}

/** One node copy's tensors: trainable weights and (for batch norm) running statistics. */
interface Params {
  w: Record<string, tf.Variable>;
  state: Record<string, tf.Variable>;
}

const isTrue = (value: ArgValue | undefined) => value === true || value === "true" || value === 1 || value === "1";

// tf.js keeps one global registry of variable names, so two models built from
// the same graph (or the same model built twice) need distinct names.
let buildCounter = 0;

export function buildGraph(model: Model, options: GraphOptions = {}): GraphModel {
  const analysis = analyze(model);
  if (analysis.errors.length) {
    const first = analysis.errors[0];
    throw new GraphError(`the model has a problem at "${first.node}": ${first.message}`);
  }
  const inputs = model.nodes.filter((node) => node.op === "input");
  if (!inputs.length) throw new GraphError("the model has no input");
  const seed = options.seed ?? 0;
  const shapeOf = (id: string) => analysis.info.get(id)!.shape;
  const prefix = `g${++buildCounter}`;

  // -- Unroll repeated blocks into a flat list of steps ------------------------
  // Same walk as toTf: a block's nodes are listed once but run `repeat` times,
  // each copy reading the previous copy's output and holding its own weights.
  const steps: Step[] = [];
  const latest = new Map<string, string>();
  const emit = (nodes: Node[], scope: string | undefined, suffix: string, alias: Map<string, string>) => {
    const read = (id: string) => {
      const key = alias.get(id) ?? latest.get(id);
      if (!key) throw new GraphError(`"${id}" is read before it is built`);
      return key;
    };
    let i = 0;
    while (i < nodes.length) {
      const node = nodes[i];
      const repeated = outermostRepeated(model.groups, node.group, scope);
      if (!repeated) {
        const key = `${node.id}${suffix}`;
        steps.push({ key, node, inputs: node.inputs.map(read), suffix });
        latest.set(node.id, key);
        i++;
        continue;
      }
      const members: Node[] = [];
      while (i < nodes.length && isWithin(model.groups, nodes[i].group, repeated.id)) members.push(nodes[i++]);
      const inner = new Set(members.map((member) => member.id));
      // The block's input is its first outside read, as in analyze and the PyTorch export.
      const entry = members.flatMap((member) => member.inputs).find((id) => !inner.has(id));
      const exit = members[members.length - 1];
      for (let copy = 0; copy < (repeated.repeat ?? 1); copy++) {
        const copyAlias = new Map(alias);
        if (copy > 0 && entry) copyAlias.set(entry, latest.get(exit.id)!);
        emit(members, repeated.id, `${suffix}_r${copy}`, copyAlias);
      }
    }
  };
  emit(model.nodes, undefined, "", new Map());

  // -- Parameters ---------------------------------------------------------------
  const params = new Map<string, Params>();
  const trainable: tf.Variable[] = [];
  const nodeOfVariable = new Map<string, string>();
  // Tied tables (a second embedding, a tied output layer) reuse the first embedding that has its own table.
  const tableNode = model.nodes.find((node) => node.op === "embedding" && !isTrue(node.args.tied));
  let table: tf.Variable | undefined;
  const hasTiedHead = model.nodes.some((node) => node.op === "lmhead" && isTrue(node.args.tied));

  for (const step of steps) {
    const { node, suffix } = step;
    const p: Params = { w: {}, state: {} };
    const inShape = node.inputs.length ? shapeOf(node.inputs[0]) : [];
    const a = node.args;
    const num = (key: string, fallback?: number) => {
      const value = a[key];
      if (value === undefined || value === "") {
        if (fallback === undefined) throw new GraphError(`${node.id} needs ${key}`);
        return fallback;
      }
      return Number(value);
    };
    const seedOf = (part: string) => hashSeed(seed, `${node.id}${suffix}${part}`);
    const add = (part: string, init: tf.Tensor) => {
      const variable = tf.variable(init, true, `${prefix}/${step.key}/${part || "w"}`);
      init.dispose();
      p.w[part] = variable;
      trainable.push(variable);
      nodeOfVariable.set(variable.name, node.id);
      return variable;
    };
    // PyTorch's default for linear and conv weights: uniform in ±1/√fan_in.
    const uniform = (part: string, shape: number[], fanIn: number) => {
      const bound = 1 / Math.sqrt(fanIn);
      return add(part, tf.randomUniform(shape, -bound, bound, "float32", seedOf(part)));
    };
    const normal = (part: string, shape: number[], std: number) => add(part, tf.randomNormal(shape, 0, std, "float32", seedOf(part)));
    const zeros = (part: string, shape: number[]) => add(part, tf.zeros(shape));
    const ones = (part: string, shape: number[]) => add(part, tf.ones(shape));
    const features = inShape[inShape.length - 1];

    switch (node.op) {
      case "conv2d": {
        const c = inShape[0];
        const out = num("out");
        const k = num("k", 3);
        const groups = a.groups === "depthwise" ? c : num("groups", 1);
        const fanIn = (c / groups) * k * k;
        // Depthwise is stored as tf.depthwiseConv2d wants it: [k, k, C, out/C].
        if (groups === c && groups > 1) uniform("", [k, k, c, out / c], fanIn);
        else uniform("", [k, k, c / groups, out], fanIn);
        if (isTrue(a.bias ?? true)) zeros("_bias", [out]);
        break;
      }
      case "linear": {
        const out = num("out");
        uniform("", [features, out], features);
        if (isTrue(a.bias ?? true)) zeros("_bias", [out]);
        break;
      }
      case "batchnorm": {
        const c = inShape[0];
        ones("_gamma", [c]);
        zeros("_beta", [c]);
        // Running statistics are state, not parameters, matching PyTorch and the engine's counts.
        p.state.mean = tf.variable(tf.zeros([c]), false, `${prefix}/${step.key}/running_mean`);
        p.state.variance = tf.variable(tf.ones([c]), false, `${prefix}/${step.key}/running_var`);
        break;
      }
      case "layernorm":
        ones("_gamma", [features]);
        if (isTrue(a.bias ?? true)) zeros("_beta", [features]);
        break;
      case "rmsnorm":
        ones("_gamma", [features]);
        break;
      case "embedding": {
        if (isTrue(a.tied)) break;
        const vocab = num("vocab");
        const dim = num("dim");
        // PyTorch draws embeddings from N(0, 1). When the output layer reuses
        // the table, that makes the first logits huge, so the table is scaled
        // to unit-length rows instead (GPT-2 does something similar).
        const variable = normal("", [vocab, dim], hasTiedHead ? 1 / Math.sqrt(dim) : 1);
        if (node.id === tableNode?.id) table = variable;
        break;
      }
      case "posembed": {
        if (isTrue(a.fixed)) break;
        const [seq, dim] = inShape;
        normal("", [num("max", seq), dim], 0.02);
        break;
      }
      case "cls_token":
        normal("", [inShape[1]], 0.02);
        break;
      case "attention": {
        const d = inShape[1];
        const heads = num("heads");
        const kv = num("kv", heads);
        const hd = num("head_dim", Math.floor(d / heads));
        uniform("_q", [d, heads * hd], d);
        uniform("_k", [d, kv * hd], d);
        uniform("_v", [d, kv * hd], d);
        uniform("_o", [heads * hd, d], heads * hd);
        if (isTrue(a.qkv_bias)) {
          zeros("_q_bias", [heads * hd]);
          zeros("_k_bias", [kv * hd]);
          zeros("_v_bias", [kv * hd]);
        }
        if (isTrue(a.o_bias)) zeros("_o_bias", [d]);
        if (isTrue(a.qk_norm)) {
          ones("_q_norm", [hd]);
          ones("_k_norm", [hd]);
        }
        break;
      }
      case "cross_attention": {
        const d = inShape[1];
        const dm = shapeOf(node.inputs[1])[1];
        const heads = num("heads");
        const kv = num("kv", heads);
        const hd = num("head_dim", Math.floor(d / heads));
        uniform("_q", [d, heads * hd], d);
        uniform("_k", [dm, kv * hd], dm);
        uniform("_v", [dm, kv * hd], dm);
        uniform("_o", [heads * hd, d], heads * hd);
        if (isTrue(a.bias)) {
          zeros("_q_bias", [heads * hd]);
          zeros("_k_bias", [kv * hd]);
          zeros("_v_bias", [kv * hd]);
          zeros("_o_bias", [d]);
        }
        break;
      }
      case "mla": {
        const d = inShape[1];
        const heads = num("heads");
        const qRank = num("q_rank", 0);
        const kvRank = num("kv_rank");
        const nope = num("qk_nope_dim");
        const rope = num("qk_rope_dim");
        const v = num("v_dim");
        if (qRank > 0) {
          uniform("_q_down", [d, qRank], d);
          ones("_q_norm", [qRank]);
          uniform("_q_up", [qRank, heads * (nope + rope)], qRank);
        } else uniform("_q", [d, heads * (nope + rope)], d);
        uniform("_kv_down", [d, kvRank + rope], d);
        ones("_kv_norm", [kvRank]);
        uniform("_kv_up", [kvRank, heads * (nope + v)], kvRank);
        uniform("_o", [heads * v, d], heads * v);
        break;
      }
      case "mlp": {
        const d = features;
        const ff = num("ff");
        const act = String(a.act ?? "swiglu");
        const gated = act !== "gelu" && act !== "relu";
        uniform("_up", [d, ff], d);
        if (gated) uniform("_gate", [d, ff], d);
        uniform("_down", [ff, d], ff);
        // The engine counts ff + d biases (up and down), so a gated layer with
        // bias=true keeps its gate unbiased to hold exactly that many.
        if (isTrue(a.bias)) {
          zeros("_up_bias", [ff]);
          zeros("_down_bias", [d]);
        }
        break;
      }
      case "moe": {
        const d = features;
        const experts = num("experts");
        const ff = num("ff");
        const shared = num("shared", 0);
        const sharedFf = num("shared_ff", ff) || ff;
        uniform("_router", [d, experts], d);
        if (isTrue(a.router_bias)) zeros("_router_bias", [experts]);
        // All experts' matrices side by side, so every expert runs in one
        // matmul. Same number of values as separate experts.
        uniform("_up", [d, experts * ff], d);
        uniform("_gate", [d, experts * ff], d);
        uniform("_down", [experts, ff, d], ff);
        for (let s = 0; s < shared; s++) {
          uniform(`_shared${s}_up`, [d, sharedFf], d);
          uniform(`_shared${s}_gate`, [d, sharedFf], d);
          uniform(`_shared${s}_down`, [sharedFf, d], sharedFf);
        }
        if (isTrue(a.shared_gate)) uniform("_shared_gate", [d, 1], d);
        break;
      }
      case "lmhead": {
        if (isTrue(a.tied)) break;
        const vocab = num("vocab");
        uniform("", [inShape[1], vocab], inShape[1]);
        break;
      }
    }
    params.set(step.key, p);
  }

  const paramCount = trainable.reduce((sum, v) => sum + v.size, 0);
  const outputNode = model.nodes[model.nodes.length - 1];
  const inputShapes = Object.fromEntries(inputs.map((node) => [node.id, shapeOf(node.id)]));

  const apply = (feeds: Record<string, tf.Tensor>, run: ApplyOptions): tf.Tensor => {
    const values = new Map<string, tf.Tensor>();
    let output: tf.Tensor | undefined;
    for (const step of steps) {
      const xs = step.inputs.map((key) => values.get(key)!);
      const value = forward(step, xs, params.get(step.key)!, feeds, run);
      values.set(step.key, value);
      output = value;
    }
    return output!;
  };

  const trace = (feeds: Record<string, tf.Tensor>): TraceStep[] => {
    const values = new Map<string, tf.Tensor>();
    const out: TraceStep[] = [];
    for (const step of steps) {
      const xs = step.inputs.map((key) => values.get(key)!);
      const kept = tf.tidy(() => {
        attentionSink = [];
        try {
          const value = forward(step, xs, params.get(step.key)!, feeds, { training: false });
          const attention = attentionSink[0];
          return attention ? [value, attention] : [value];
        } finally {
          attentionSink = null;
        }
      });
      values.set(step.key, kept[0]);
      const copy = Number(/_r(\d+)/.exec(step.suffix)?.[1] ?? 0);
      out.push({ key: step.key, nodeId: step.node.id, copy, value: kept[0], attention: kept[1] });
    }
    return out;
  };

  const forward = (step: Step, xs: tf.Tensor[], p: Params, feeds: Record<string, tf.Tensor>, run: ApplyOptions): tf.Tensor => {
    const { node } = step;
    const a = node.args;
    const num = (key: string, fallback: number) => (a[key] === undefined || a[key] === "" ? fallback : Number(a[key]));
    const inShape = node.inputs.length ? shapeOf(node.inputs[0]) : [];
    const engineRank = inShape.length;
    const [x] = xs;
    const w = p.w;

    switch (node.op) {
      case "input": {
        const fed = feeds[node.id];
        if (!fed) throw new GraphError(`no data fed to input "${node.id}"`);
        return fed;
      }
      case "identity":
        return x;

      // -- Images -------------------------------------------------------------
      case "conv2d": {
        const c = inShape[0];
        const k = num("k", 3);
        const s = num("s", 1);
        const pad = num("p", Math.floor(k / 2));
        const groups = a.groups === "depthwise" ? c : num("groups", 1);
        // PyTorch pads evenly on both sides; pad explicitly and convolve "valid".
        const padded = (pad > 0 ? tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]) : x) as tf.Tensor4D;
        let y: tf.Tensor4D;
        if (groups === 1) y = tf.conv2d(padded, w[""] as tf.Tensor4D, s, "valid");
        else if (groups === c) y = tf.depthwiseConv2d(padded, w[""] as tf.Tensor4D, s, "valid");
        else {
          // Grouped: each slice of input channels has its own slice of filters.
          const parts = tf.split(padded, groups, 3);
          const filters = tf.split(w[""], groups, 3);
          y = tf.concat(parts.map((part, g) => tf.conv2d(part as tf.Tensor4D, filters[g] as tf.Tensor4D, s, "valid") as tf.Tensor4D), 3);
        }
        return w._bias ? tf.add(y, w._bias) : y;
      }
      case "maxpool": {
        const k = num("k", 2);
        const pad = num("p", 0);
        // A very negative border never wins the max, as in PyTorch.
        const padded = (pad > 0 ? tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], -1e30) : x) as tf.Tensor4D;
        return tf.maxPool(padded, k, num("s", k), "valid");
      }
      case "avgpool": {
        const k = num("k", 2);
        const pad = num("p", 0);
        // PyTorch counts zero padding in the average by default.
        const padded = (pad > 0 ? tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]) : x) as tf.Tensor4D;
        return tf.avgPool(padded, k, num("s", k), "valid");
      }
      case "gap":
        // Kept as [1, 1, C] so a gate can multiply it back onto the image.
        return tf.mean(x, [1, 2], true);
      case "batchnorm": {
        // The engine's channel axis is 0: channels for images, positions for sequences.
        const tfRank = x.rank;
        const channelAxis = engineRank === 3 ? 3 : 1;
        const axes = Array.from({ length: tfRank }, (_, i) => i).filter((i) => i !== channelAxis);
        const shape = Array.from({ length: tfRank }, (_, i) => (i === channelAxis ? inShape[0] : 1));
        let mean: tf.Tensor;
        let variance: tf.Tensor;
        if (run.training) {
          const moments = tf.moments(x, axes);
          mean = moments.mean;
          variance = moments.variance;
          // Running averages for evaluation, PyTorch style: momentum 0.1, unbiased variance.
          const n = x.size / inShape[0];
          const unbiased = tf.mul(variance, n / Math.max(1, n - 1));
          p.state.mean.assign(tf.add(tf.mul(p.state.mean, 0.9), tf.mul(mean, 0.1)));
          p.state.variance.assign(tf.add(tf.mul(p.state.variance, 0.9), tf.mul(unbiased, 0.1)));
        } else {
          mean = p.state.mean;
          variance = p.state.variance;
        }
        const normed = tf.mul(tf.sub(x, tf.reshape(mean, shape)), tf.reshape(tf.rsqrt(tf.add(variance, 1e-5)), shape));
        return tf.add(tf.mul(normed, tf.reshape(w._gamma, shape)), tf.reshape(w._beta, shape));
      }
      case "to_sequence": {
        // [B, H, W, C] to [B, H*W, C]; row-major positions, as PyTorch's flatten(2).transpose(1, 2).
        const [c, h, wd] = inShape;
        return tf.reshape(x, [-1, h * wd, c]);
      }

      // -- Dense, norms and shape ----------------------------------------------
      case "linear":
        return onEngineLast(x, engineRank, (t) => dense(t, w[""], w._bias));
      case "layernorm":
        return onEngineLast(x, engineRank, (t) => {
          const { mean, variance } = tf.moments(t, -1, true);
          const normed = tf.mul(tf.sub(t, mean), tf.rsqrt(tf.add(variance, 1e-5)));
          const scaled = tf.mul(normed, w._gamma);
          return w._beta ? tf.add(scaled, w._beta) : scaled;
        });
      case "rmsnorm":
        return onEngineLast(x, engineRank, (t) => rmsNorm(t, w._gamma));
      case "flatten":
        return engineRank <= 1 ? x : tf.reshape(x, [x.shape[0], -1]);
      case "dropout": {
        const rate = num("rate", 0.5);
        if (!run.training || rate <= 0) return x;
        return tf.dropout(x, rate, undefined, hashSeed(seed, `${step.key}#${run.step ?? 0}`));
      }

      // -- Activations -----------------------------------------------------------
      case "relu":
        return tf.relu(x);
      case "relu6":
        return tf.relu6(x);
      case "gelu":
        return gelu(x);
      case "silu":
        return silu(x);
      case "sigmoid":
        return tf.sigmoid(x);
      case "tanh":
        return tf.tanh(x);
      case "hardswish":
        return tf.div(tf.mul(x, tf.relu6(tf.add(x, 3))), 6);
      case "softmax":
        return onEngineLast(x, engineRank, (t) => tf.softmax(t, -1));

      // -- Joins -----------------------------------------------------------------
      case "add":
        return xs.slice(1).reduce((sum, t) => tf.add(sum, t), x);
      case "mul":
        // tf broadcasts a [1, 1, C] gate over [H, W, C], like [C, 1, 1] over [C, H, W].
        return xs.slice(1).reduce((product, t) => tf.mul(product, t), x);
      case "concat": {
        let axis = num("axis", engineRank === 3 ? 0 : engineRank - 1);
        if (axis < 0) axis += engineRank;
        const tfAxis = engineRank === 3 ? [3, 1, 2][axis] : axis + 1;
        return tf.concat(xs, tfAxis);
      }

      // -- Sequences -------------------------------------------------------------
      case "embedding": {
        const source = isTrue(a.tied) ? table : w[""];
        if (!source) throw new GraphError(`embedding "${node.id}" is tied but the model has no embedding with its own table`);
        return tf.gather(source, tf.cast(x, "int32"));
      }
      case "posembed": {
        const [seq, dim] = inShape;
        if (isTrue(a.fixed)) return tf.add(x, sineTable(seq, dim));
        // A token-type table: every token is segment 0, so its first row is added to all.
        if (isTrue(a.types)) return tf.add(x, tf.slice(w[""], [0, 0], [1, dim]));
        return tf.add(x, tf.slice(w[""], [0, 0], [seq, dim]));
      }
      case "cls_token": {
        const batch = x.shape[0];
        const token = tf.tile(tf.reshape(w[""], [1, 1, -1]), [batch, 1, 1]);
        return tf.concat([token, x], 1);
      }
      case "take_first": {
        const [, d] = inShape;
        return tf.reshape(tf.slice(x, [0, 0, 0], [-1, 1, d]), [-1, d]);
      }
      case "mean_pool":
        return tf.mean(x, 1);
      case "attention":
        return selfAttention(x, w, a, model.kind === "language");
      case "cross_attention":
        return crossAttention(x, xs[1], w, a);
      case "mla":
        return latentAttention(x, w, a);
      case "mlp": {
        const act = String(a.act ?? "swiglu");
        const gated = act !== "gelu" && act !== "relu";
        const fn = act === "relu" ? tf.relu : act === "gelu" || act === "geglu" ? gelu : silu;
        const up = dense(x, w._up, w._up_bias);
        const h = gated ? tf.mul(fn(dense(x, w._gate)), up) : fn(up);
        return dense(h, w._down, w._down_bias);
      }
      case "moe":
        return mixtureOfExperts(x, w, a);
      case "lmhead": {
        if (isTrue(a.tied)) {
          if (!table) throw new GraphError(`lmhead "${node.id}" is tied but the model has no embedding with its own table`);
          const [b, s, d] = x.shape;
          return tf.reshape(tf.matMul(tf.reshape(x, [-1, d]), table, false, true), [b, s, -1]);
        }
        return dense(x, w[""]);
      }
      default:
        throw new GraphError(`"${node.op}" is not supported by the browser trainer`);
    }
  };

  return {
    model,
    analysis,
    inputs,
    inputShapes,
    outputNode,
    outputShape: shapeOf(outputNode.id),
    endsInSoftmax: outputNode.op === "softmax",
    trainable,
    nodeOfVariable,
    paramCount,
    apply,
    trace,
    dispose() {
      for (const p of params.values()) for (const v of [...Object.values(p.w), ...Object.values(p.state)]) v.dispose();
    },
  };
}

// -- Building blocks ---------------------------------------------------------------

/** x @ W (+ b) over the last axis, for any rank. */
function dense(x: tf.Tensor, weight: tf.Tensor, bias?: tf.Tensor): tf.Tensor {
  const shape = x.shape;
  const inF = shape[shape.length - 1];
  let y: tf.Tensor = tf.matMul(tf.reshape(x, [-1, inF]), weight as tf.Tensor2D);
  if (bias) y = tf.add(y, bias);
  return tf.reshape(y, [...shape.slice(0, -1), weight.shape[weight.shape.length - 1]]);
}

/**
 * Ops the engine defines on the last axis (linear, norms, softmax) mean the
 * width axis for an image [C, H, W]. In channels-last tf that axis is not
 * last, so move to channels-first, apply, and move back.
 */
function onEngineLast(x: tf.Tensor, engineRank: number, fn: (t: tf.Tensor) => tf.Tensor): tf.Tensor {
  if (engineRank !== 3) return fn(x);
  return tf.transpose(fn(tf.transpose(x, [0, 3, 1, 2])), [0, 2, 3, 1]);
}

function rmsNorm(x: tf.Tensor, gamma: tf.Tensor, eps = 1e-6): tf.Tensor {
  return tf.mul(tf.mul(x, tf.rsqrt(tf.add(tf.mean(tf.square(x), -1, true), eps))), gamma);
}

/** Exact GELU with erf, PyTorch's default. */
const gelu = (x: tf.Tensor): tf.Tensor => tf.mul(tf.mul(x, 0.5), tf.add(1, tf.erf(tf.div(x, Math.SQRT2))));
const silu = (x: tf.Tensor): tf.Tensor => tf.mul(x, tf.sigmoid(x));

/** [B, S, H*hd] to [B, H, S, hd]. */
const splitHeads = (x: tf.Tensor, heads: number, hd: number) => {
  const [b, s] = x.shape;
  return tf.transpose(tf.reshape(x, [b, s, heads, hd]), [0, 2, 1, 3]);
};
/** [B, H, S, hd] to [B, S, H*hd]. */
const mergeHeads = (x: tf.Tensor) => {
  const [b, h, s, hd] = x.shape;
  return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [b, s, h * hd]);
};
/** Grouped-query attention: each key/value head serves `group` query heads in a row (PyTorch's enable_gqa). */
const repeatHeads = (x: tf.Tensor, group: number) => {
  if (group === 1) return x;
  // Gather rather than tile: tf.js has no gradient for tiling rank-5 tensors.
  const heads = Array.from({ length: x.shape[1]! * group }, (_, i) => Math.floor(i / group));
  return tf.gather(x, tf.tensor1d(heads, "int32"), 1);
};

/** Rotary positions on the last axis of [..., seq, dim], split in halves as the PyTorch export's apply_rope. */
function rope(x: tf.Tensor, theta = 10000): tf.Tensor {
  const seq = x.shape[x.rank - 2];
  const dim = x.shape[x.rank - 1];
  const half = Math.floor(dim / 2);
  const cos = new Float32Array(seq * half);
  const sin = new Float32Array(seq * half);
  for (let p = 0; p < seq; p++)
    for (let i = 0; i < half; i++) {
      const angle = p * theta ** (-i / half);
      cos[p * half + i] = Math.cos(angle);
      sin[p * half + i] = Math.sin(angle);
    }
  const c = tf.tensor2d(cos, [seq, half]);
  const s = tf.tensor2d(sin, [seq, half]);
  const [x1, x2] = tf.split(x, [half, dim - half], -1);
  return tf.concat([tf.sub(tf.mul(x1, c), tf.mul(x2, s)), tf.add(tf.mul(x1, s), tf.mul(x2, c))], -1);
}

/** The original Transformer's sine and cosine positions: even features sine, odd cosine. */
function sineTable(seq: number, dim: number): tf.Tensor2D {
  const table = new Float32Array(seq * dim);
  for (let p = 0; p < seq; p++)
    for (let i = 0; i < dim; i += 2) {
      const rate = Math.exp(i * (-Math.log(10000) / dim));
      table[p * dim + i] = Math.sin(p * rate);
      if (i + 1 < dim) table[p * dim + i + 1] = Math.cos(p * rate);
    }
  return tf.tensor2d(table, [seq, dim]);
}

/**
 * An additive mask [S, S]: 0 where query i may see key j, a large negative
 * number elsewhere. Causal hides the future; a sliding window also hides keys
 * more than `window` back (and, as in the PyTorch export, is always causal).
 */
function attentionMask(seq: number, causal: boolean, window: number): tf.Tensor2D | null {
  if (!causal && !window) return null;
  const mask = new Float32Array(seq * seq);
  for (let i = 0; i < seq; i++)
    for (let j = 0; j < seq; j++) {
      const allowed = j <= i && (!window || i - j < window);
      mask[i * seq + j] = allowed ? 0 : -1e9;
    }
  return tf.tensor2d(mask, [seq, seq]);
}

/** softmax(q kᵀ · scale + mask) v over [B, H, S, d]. */
function attend(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, scale: number, mask: tf.Tensor | null): tf.Tensor {
  let scores = tf.mul(tf.matMul(q as tf.Tensor4D, k as tf.Tensor4D, false, true), scale);
  if (mask) scores = tf.add(scores, mask);
  const weights = tf.softmax(scores, -1);
  if (attentionSink) attentionSink.push(weights);
  return tf.matMul(weights as tf.Tensor4D, v as tf.Tensor4D);
}

type Weights = Record<string, tf.Variable>;
type NodeArgs = Record<string, ArgValue>;
const argNum = (a: NodeArgs, key: string, fallback: number) => (a[key] === undefined || a[key] === "" ? fallback : Number(a[key]));

function selfAttention(x: tf.Tensor, w: Weights, a: NodeArgs, isLanguage: boolean): tf.Tensor {
  const [, s, d] = x.shape;
  const heads = argNum(a, "heads", 1);
  const kv = argNum(a, "kv", heads);
  const hd = argNum(a, "head_dim", Math.floor(d / heads));
  let q = splitHeads(dense(x, w._q, w._q_bias), heads, hd);
  let k = splitHeads(dense(x, w._k, w._k_bias), kv, hd);
  const v = splitHeads(dense(x, w._v, w._v_bias), kv, hd);
  if (w._q_norm) {
    q = rmsNorm(q, w._q_norm);
    k = rmsNorm(k, w._k_norm);
  }
  // Rotary positions are on by default for language models, as in the export.
  if (a.rope === undefined ? isLanguage : isTrue(a.rope)) {
    q = rope(q);
    k = rope(k);
  }
  const causal = a.causal === undefined ? true : isTrue(a.causal);
  const mask = attentionMask(s, causal, argNum(a, "window", 0));
  const out = attend(q, repeatHeads(k, heads / kv), repeatHeads(v, heads / kv), 1 / Math.sqrt(hd), mask);
  return dense(mergeHeads(out), w._o, w._o_bias);
}

function crossAttention(x: tf.Tensor, memory: tf.Tensor, w: Weights, a: NodeArgs): tf.Tensor {
  const d = x.shape[2]!;
  const heads = argNum(a, "heads", 1);
  const kv = argNum(a, "kv", heads);
  const hd = argNum(a, "head_dim", Math.floor(d / heads));
  const q = splitHeads(dense(x, w._q, w._q_bias), heads, hd);
  const k = splitHeads(dense(memory, w._k, w._k_bias), kv, hd);
  const v = splitHeads(dense(memory, w._v, w._v_bias), kv, hd);
  // Every decoder position may read every encoder position: no mask.
  const out = attend(q, repeatHeads(k, heads / kv), repeatHeads(v, heads / kv), 1 / Math.sqrt(hd), null);
  return dense(mergeHeads(out), w._o, w._o_bias);
}

/** DeepSeek's latent attention, as the export's LatentAttention. */
function latentAttention(x: tf.Tensor, w: Weights, a: NodeArgs): tf.Tensor {
  const [, s] = x.shape;
  const heads = argNum(a, "heads", 1);
  const kvRank = argNum(a, "kv_rank", 0);
  const nope = argNum(a, "qk_nope_dim", 0);
  const ropeDim = argNum(a, "qk_rope_dim", 0);
  const vDim = argNum(a, "v_dim", 0);
  const qFlat = w._q_down ? dense(rmsNorm(dense(x, w._q_down), w._q_norm), w._q_up) : dense(x, w._q);
  const q = splitHeads(qFlat, heads, nope + ropeDim);
  const [qNope, qRope] = tf.split(q, [nope, ropeDim], -1);
  const [latent, kRopeFlat] = tf.split(dense(x, w._kv_down), [kvRank, ropeDim], -1);
  const kvUp = splitHeads(dense(rmsNorm(latent, w._kv_norm), w._kv_up), heads, nope + vDim);
  const [kNope, v] = tf.split(kvUp, [nope, vDim], -1);
  // One rotary key shared by every head.
  const kRope = tf.tile(rope(tf.expandDims(kRopeFlat, 1)), [1, heads, 1, 1]);
  const qAll = tf.concat([qNope, rope(qRope)], -1);
  const kAll = tf.concat([kNope, kRope], -1);
  const causal = a.causal === undefined ? true : isTrue(a.causal);
  const out = attend(qAll, kAll, v, 1 / Math.sqrt(nope + ropeDim), attentionMask(s, causal, 0));
  return dense(mergeHeads(out), w._o);
}

/**
 * Zeroes all but the k largest values in each row. Choosing is not
 * differentiable (tf.js has no gradient for topk), so the choice is made
 * inside a custom gradient: the kept values pass gradients straight through,
 * the dropped ones get none.
 */
const keepTopK = (x: tf.Tensor, k: number): tf.Tensor =>
  tf.customGrad((...args: (tf.Tensor | tf.GradSaveFunc)[]) => {
    const t = args[0] as tf.Tensor;
    const save = args[1] as tf.GradSaveFunc;
    const mask = tf.sum(tf.oneHot(tf.topk(t, k).indices, t.shape[t.rank - 1]), 1).toFloat();
    save([mask]);
    return { value: tf.mul(t, mask), gradFunc: (dy, saved) => tf.mul(dy as tf.Tensor, saved[0]) };
  })(x);

/**
 * Mixture of experts as the export's MixtureOfExperts: router softmax, top-k,
 * weights renormalised over the chosen experts. At browser sizes it is
 * simpler and faster to run every expert on every token and zero the ones not
 * chosen than to gather tokens per expert; the result is the same.
 */
function mixtureOfExperts(x: tf.Tensor, w: Weights, a: NodeArgs): tf.Tensor {
  const shape = x.shape;
  const d = shape[shape.length - 1];
  const experts = argNum(a, "experts", 1);
  const topk = argNum(a, "topk", 2);
  const ff = argNum(a, "ff", 0);
  const shared = argNum(a, "shared", 0);
  const flat = tf.reshape(x, [-1, d]);
  const n = flat.shape[0];
  let scores = tf.matMul(flat, w._router as tf.Tensor2D);
  if (w._router_bias) scores = tf.add(scores, w._router_bias);
  const probs = tf.softmax(scores, -1);
  const picked = keepTopK(probs, topk);
  const weights = tf.div(picked, tf.sum(picked, -1, true));
  // Every expert at once: [N, E*ff] then [E, N, ff] @ [E, ff, d].
  const up = tf.reshape(tf.matMul(flat, w._up as tf.Tensor2D), [n, experts, ff]);
  const gate = tf.reshape(tf.matMul(flat, w._gate as tf.Tensor2D), [n, experts, ff]);
  const hidden = tf.transpose(tf.mul(silu(gate), up), [1, 0, 2]);
  const outs = tf.matMul(hidden as tf.Tensor3D, w._down as tf.Variable<tf.Rank.R3>);
  let out: tf.Tensor = tf.sum(tf.mul(outs, tf.expandDims(tf.transpose(weights), 2)), 0);
  for (let s = 0; s < shared; s++) {
    const h = tf.mul(silu(tf.matMul(flat, w[`_shared${s}_gate`] as tf.Tensor2D)), tf.matMul(flat, w[`_shared${s}_up`] as tf.Tensor2D));
    const extra = tf.matMul(h, w[`_shared${s}_down`] as tf.Tensor2D);
    out = tf.add(out, w._shared_gate ? tf.mul(tf.sigmoid(tf.matMul(flat, w._shared_gate as tf.Tensor2D)), extra) : extra);
  }
  return tf.reshape(out, shape);
}
