import * as tf from "@tensorflow/tfjs";
import { analyze, type Group, type Model, type Node, type Shape } from "tessera-engine";
import { ConstantPad2d, HardSwish } from "./layers";

// Turns an engine graph into a tf.js functional model that can be trained in
// the browser. The engine works channels-first ([C, H, W], as PyTorch does);
// tf.js layers are channels-last, so images become [H, W, C] here. Shapes,
// parameter counts and the maths match; only the order a flatten lays values
// out in differs, which does not matter for training from scratch.

/** Ops that only make sense for language models; training those needs a GPU, so they go to the Colab export. */
const LANGUAGE_OPS = new Set(["embedding", "posembed", "attention", "mla", "mlp", "moe", "lmhead", "rmsnorm"]);

export class UnsupportedOpError extends Error {}

export interface TfBuild {
  model: tf.LayersModel;
  /** Which engine node each tf.js layer came from. Unrolled copies of a repeated block map to the same node. */
  layerToNode: Map<string, string>;
  /** True when the graph ends in its own softmax, so the output is probabilities rather than logits. */
  endsInSoftmax: boolean;
  /** The engine's input shape, channels first. */
  inputShape: Shape;
  /** The engine's output shape. */
  outputShape: Shape;
}

export interface TfOptions {
  /** Seeds the initial weights. The same seed gives the same weights to layers with the same id, so two variants of a model start alike. */
  seed?: number;
}

/** Builds a tf.js model from the graph. Throws for ops that cannot be trained in the browser. */
export function toTf(model: Model, options: TfOptions = {}): tf.LayersModel {
  return buildTf(model, options).model;
}

/** Trainable parameters only, which is what the engine and PyTorch count (batch norm's running statistics are left out). */
export function trainableParams(model: tf.LayersModel): number {
  return model.trainableWeights.reduce((sum, w) => sum + w.shape.reduce<number>((a, b) => a * (b ?? 1), 1), 0);
}

export function buildTf(model: Model, options: TfOptions = {}): TfBuild {
  for (const node of model.nodes) {
    if (LANGUAGE_OPS.has(node.op))
      throw new UnsupportedOpError(`"${node.op}" (${node.label ?? node.id}) is a language model op and cannot be trained in the browser. Use the Colab export to train this model.`);
  }
  const analysis = analyze(model);
  if (analysis.errors.length) {
    const first = analysis.errors[0];
    throw new Error(`the model has a problem at "${first.node}": ${first.message}`);
  }
  const inputs = model.nodes.filter((node) => node.op === "input");
  if (inputs.length !== 1) throw new UnsupportedOpError(`training in the browser needs exactly one input, this model has ${inputs.length}`);

  const seed = options.seed ?? 0;
  const shapeOf = (id: string) => analysis.info.get(id)!.shape;
  const values = new Map<string, tf.SymbolicTensor>();
  const layerToNode = new Map<string, string>();
  const usedNames = new Set<string>();
  let input: tf.SymbolicTensor | undefined;

  const layerName = (node: Node, suffix: string, part = "") => {
    const base = `${node.id}${suffix}${part}`.replace(/[^A-Za-z0-9_.-]/g, "_");
    let name = base;
    for (let n = 2; usedNames.has(name); n++) name = `${base}_${n}`;
    usedNames.add(name);
    layerToNode.set(name, node.id);
    return name;
  };

  const build = (node: Node, suffix: string, read: (id: string) => tf.SymbolicTensor) => {
    const name = (part = "") => layerName(node, suffix, part);
    const xs = node.inputs.map(read);
    const inShape = node.inputs.length ? shapeOf(node.inputs[0]) : [];
    const x = xs[0];
    const apply = (layer: tf.layers.Layer, value: tf.SymbolicTensor | tf.SymbolicTensor[] = x) => layer.apply(value) as tf.SymbolicTensor;
    const args = node.args;
    const num = (key: string, fallback: number) => (args[key] === undefined || args[key] === "" ? fallback : Number(args[key]));
    const flag = (key: string, fallback: boolean) => (args[key] === undefined ? fallback : args[key] === true || args[key] === "true" || args[key] === 1 || args[key] === "1");
    const layerSeed = (part: string) => hashSeed(seed, `${node.id}${suffix}${part}`);
    const rank = inShape.length;

    switch (node.op) {
      case "input": {
        const shape = shapeOf(node.id);
        input = tf.input({ shape: shape.length === 3 ? toChannelsLast(shape) : shape, name: name() });
        return input;
      }
      case "conv2d": {
        const [c] = inShape;
        const out = num("out", NaN);
        const k = num("k", 3);
        const s = num("s", 1);
        const p = num("p", Math.floor(k / 2));
        const groups = args.groups === "depthwise" ? c : num("groups", 1);
        const useBias = flag("bias", true);
        // PyTorch pads explicitly and evenly; tf.js "same" pads unevenly on
        // strided layers, so pad first and convolve without padding.
        const padded = p > 0 ? apply(tf.layers.zeroPadding2d({ padding: p, name: name("_pad") })) : x;
        if (groups === 1) {
          return apply(
            tf.layers.conv2d({ filters: out, kernelSize: k, strides: s, padding: "valid", useBias, kernelInitializer: torchInit(layerSeed("")), name: name() }),
            padded,
          );
        }
        if (groups === c) {
          // Each input channel gets its own filters: a depthwise convolution.
          // PyTorch's fan-in here is k*k, not c*k*k.
          const bound = 1 / Math.sqrt(k * k);
          return apply(
            tf.layers.depthwiseConv2d({ kernelSize: k, depthMultiplier: out / c, strides: s, padding: "valid", useBias, depthwiseInitializer: tf.initializers.randomUniform({ minval: -bound, maxval: bound, seed: layerSeed("") }), name: name() }),
            padded,
          );
        }
        throw new UnsupportedOpError(`conv2d "${node.label ?? node.id}" uses ${groups} groups; grouped convolutions other than depthwise are not supported for training in the browser. Use the Colab export.`);
      }
      case "maxpool": {
        const k = num("k", 2);
        const p = num("p", 0);
        const padded = p > 0 ? apply(new ConstantPad2d({ pad: p, value: -1e30, name: name("_pad") })) : x;
        return apply(tf.layers.maxPooling2d({ poolSize: k, strides: num("s", k), padding: "valid", name: name() }), padded);
      }
      case "avgpool": {
        const k = num("k", 2);
        const p = num("p", 0);
        // PyTorch counts the zero padding in the average by default, so zero padding matches.
        const padded = p > 0 ? apply(tf.layers.zeroPadding2d({ padding: p, name: name("_pad") })) : x;
        return apply(tf.layers.averagePooling2d({ poolSize: k, strides: num("s", k), padding: "valid", name: name() }), padded);
      }
      case "gap": {
        // The engine keeps [C, 1, 1] so a gate can multiply it back onto the image.
        const pooled = apply(tf.layers.globalAveragePooling2d({ name: name() }));
        return apply(tf.layers.reshape({ targetShape: [1, 1, inShape[0]], name: name("_keep") }), pooled);
      }
      case "flatten":
        return rank <= 1 ? x : apply(tf.layers.flatten({ name: name() }));
      case "linear": {
        if (rank === 3)
          throw new UnsupportedOpError(`linear "${node.label ?? node.id}" reads an image directly; add a flatten (or gap and flatten) before it to train in the browser`);
        return apply(tf.layers.dense({ units: num("out", NaN), useBias: flag("bias", true), kernelInitializer: torchInit(layerSeed("")), name: name() }));
      }
      case "batchnorm":
        // PyTorch normalises dimension 1 (channels); channels are last in tf.js
        // for images, and stay at axis 1 for sequences.
        return apply(tf.layers.batchNormalization({ axis: rank === 2 ? 1 : -1, epsilon: 1e-5, momentum: 0.9, name: name() }));
      case "layernorm":
        if (rank === 3) throw new UnsupportedOpError(`layernorm on an image is not supported for training in the browser`);
        return apply(tf.layers.layerNormalization({ axis: -1, epsilon: 1e-5, center: flag("bias", true), name: name() }));
      case "dropout":
        return apply(tf.layers.dropout({ rate: num("rate", 0.5), name: name() }));
      case "relu":
      case "relu6":
      case "gelu":
      case "sigmoid":
      case "tanh":
      case "softmax":
        return apply(tf.layers.activation({ activation: node.op, name: name() }));
      case "silu":
        return apply(tf.layers.activation({ activation: "swish", name: name() }));
      case "hardswish":
        return apply(new HardSwish({ name: name() }));
      case "add":
        return apply(tf.layers.add({ name: name() }), xs);
      case "mul":
        // tf.js broadcasts a [1, 1, C] gate over [H, W, C].
        return apply(tf.layers.multiply({ name: name() }), xs);
      case "concat": {
        let axis = num("axis", rank === 3 ? 0 : rank - 1);
        if (axis < 0) axis += rank;
        const tfAxis = rank === 3 ? [3, 1, 2][axis] : axis + 1;
        return apply(tf.layers.concatenate({ axis: tfAxis, name: name() }), xs);
      }
      default:
        throw new UnsupportedOpError(`"${node.op}" is not supported for training in the browser`);
    }
  };

  // Repeated blocks list their nodes once; train needs every copy. Emit a
  // block's nodes `repeat` times, feeding each copy's output to the next.
  const emit = (nodes: Node[], scope: string | undefined, suffix: string, alias: Map<string, tf.SymbolicTensor>) => {
    const read = (id: string) => {
      const value = alias.get(id) ?? values.get(id);
      if (!value) throw new Error(`"${id}" is read before it is built`);
      return value;
    };
    let i = 0;
    while (i < nodes.length) {
      const node = nodes[i];
      const repeated = outermostRepeated(model.groups, node.group, scope);
      if (!repeated) {
        values.set(node.id, build(node, suffix, read));
        i++;
        continue;
      }
      const members: Node[] = [];
      while (i < nodes.length && isWithin(model.groups, nodes[i].group, repeated.id)) members.push(nodes[i++]);
      const inner = new Set(members.map((member) => member.id));
      const entry = members.flatMap((member) => member.inputs).find((id) => !inner.has(id));
      const exit = members[members.length - 1];
      for (let copy = 0; copy < (repeated.repeat ?? 1); copy++) {
        const copyAlias = new Map(alias);
        if (copy > 0 && entry) copyAlias.set(entry, values.get(exit.id)!);
        emit(members, repeated.id, `${suffix}_r${copy}`, copyAlias);
      }
    }
  };
  emit(model.nodes, undefined, "", new Map());

  const last = model.nodes[model.nodes.length - 1];
  const output = values.get(last.id)!;
  const tfModel = tf.model({ inputs: input!, outputs: output, name: model.name.replace(/[^A-Za-z0-9_.-]/g, "_") });
  return { model: tfModel, layerToNode, endsInSoftmax: last.op === "softmax", inputShape: shapeOf(inputs[0].id), outputShape: shapeOf(last.id) };
}

/** [C, H, W] to [H, W, C]. */
export const toChannelsLast = ([c, h, w]: Shape): Shape => [h, w, c];

/** The repeated block (repeat > 1) furthest out that holds `group` and sits inside `scope`. */
export function outermostRepeated(groups: Group[], group: string | undefined, scope: string | undefined): Group | undefined {
  let found: Group | undefined;
  const seen = new Set<string>();
  let id = group;
  while (id && id !== scope && !seen.has(id)) {
    seen.add(id);
    const g = groups.find((item) => item.id === id);
    if (!g) break;
    if ((g.repeat ?? 1) > 1) found = g;
    id = g.parent;
  }
  return found;
}

export function isWithin(groups: Group[], group: string | undefined, ancestor: string): boolean {
  const seen = new Set<string>();
  let id = group;
  while (id && !seen.has(id)) {
    if (id === ancestor) return true;
    seen.add(id);
    id = groups.find((item) => item.id === id)?.parent;
  }
  return false;
}

/**
 * PyTorch's default for conv and linear weights (kaiming uniform with a = √5),
 * which works out to uniform in ±1/√fan_in. Matching it keeps browser runs
 * comparable with the Colab export.
 */
const torchInit = (seed: number) => tf.initializers.varianceScaling({ scale: 1 / 3, mode: "fanIn", distribution: "uniform", seed });

/** A stable 31-bit seed from a run seed and a layer's name (FNV-1a). */
export function hashSeed(seed: number, key: string): number {
  let h = 2166136261 ^ seed;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 2147483647;
}

/** Why a model cannot be trained in the browser, or null if it can. Builds the model once to find out, then frees it. */
export function trainingProblem(model: Model): string | null {
  try {
    buildTf(model).model.dispose();
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}
