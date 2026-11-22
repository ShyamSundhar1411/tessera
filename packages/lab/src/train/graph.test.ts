import { describe, expect, test } from "bun:test";
import * as tf from "@tensorflow/tfjs";
import { analyze, Builder, CATALOG, catalogModel, shrinkLanguageModel, type ArgValue, type Model } from "tessera-engine";
import { buildGraph, type GraphModel } from "./graph";
import { prepareForTraining } from "./tasks";

await tf.setBackend("cpu");

/** Zeros (or given values) for every input, shaped as the graph expects, channels-last for images. */
function feedsFor(graph: GraphModel, batch = 2, fill?: (id: string, shape: number[]) => tf.Tensor) {
  const feeds: Record<string, tf.Tensor> = {};
  for (const input of graph.inputs) {
    const shape = graph.inputShapes[input.id];
    const tokens = graph.model.nodes.some((node) => node.op === "embedding" && node.inputs.includes(input.id));
    if (fill) feeds[input.id] = fill(input.id, shape);
    else if (tokens) feeds[input.id] = tf.zeros([batch, ...shape], "int32");
    else feeds[input.id] = tf.randomUniform([batch, ...(shape.length === 3 ? [shape[1], shape[2], shape[0]] : shape)], 0, 1, "float32", 3);
  }
  return feeds;
}

const tfShape = (engine: number[]) => (engine.length === 3 ? [engine[1], engine[2], engine[0]] : engine);

// The strongest check that the executor is right: it must hold exactly as many
// trainable values as the engine counts, for every kind of model.
describe("graph executor matches the engine's parameter counts", () => {
  const shrinkable = CATALOG.filter((entry) => {
    const model = entry.build();
    return model.kind === "language" || model.nodes.some((node) => node.op === "to_sequence");
  });
  for (const entry of shrinkable) {
    test(`${entry.id} (shrunk)`, () => {
      const model = shrinkLanguageModel(entry.build(), { dim: 64, layers: 3, vocab: 97, seqLen: 16 });
      const graph = buildGraph(model);
      expect(graph.paramCount).toBe(analyze(model).params);
      const out = tf.tidy(() => graph.apply(feedsFor(graph), { training: false }).shape);
      expect(out.slice(1)).toEqual(tfShape(analyze(model).outputShape!));
      graph.dispose();
    });
  }
  for (const id of ["mnist-mlp", "lenet5", "cifar-cnn-skip", "mobilenet-v2"]) {
    test(id, () => {
      const model = catalogModel(id)!;
      const graph = buildGraph(model);
      expect(graph.paramCount).toBe(analyze(model).params);
      graph.dispose();
    });
  }
  test("every model prepared for training", () => {
    for (const entry of CATALOG) {
      const prepared = prepareForTraining(entry.build());
      if (!prepared.ok) throw new Error(`${entry.id}: ${prepared.reason}`);
      // Skip building the big CNNs here; their counts are checked above and in toTf's tests.
      if (prepared.params > 3e6) continue;
      const graph = buildGraph(prepared.model);
      expect(graph.paramCount).toBe(prepared.params);
      graph.dispose();
    }
  });
});

/** A one-op model around `op`, reading a sequence [seq, dim] (or tokens / an image when asked). */
function around(op: string, args: Record<string, ArgValue>, input = "8x16"): Model {
  const b = new Builder(`one ${op}`, "language");
  let x = b.input(input);
  if (input.split("x").length === 1) x = b.node("embedding", x, { vocab: 20, dim: 16 });
  b.node(op, x, args);
  return b.build();
}

describe("every op runs with the shape the engine infers", () => {
  const cases: [string, Model][] = [
    ["attention causal, grouped-query, rope, qk norm, biases", around("attention", { heads: 4, kv: 2, qk_norm: true, qkv_bias: true, o_bias: true, rope: true })],
    ["attention bidirectional", around("attention", { heads: 2, causal: false, rope: false })],
    ["attention sliding window", around("attention", { heads: 2, window: 3 })],
    ["mla with a query latent", around("mla", { heads: 2, q_rank: 8, kv_rank: 8, qk_nope_dim: 4, qk_rope_dim: 4, v_dim: 4 })],
    ["mla without a query latent", around("mla", { heads: 2, kv_rank: 8, qk_nope_dim: 4, qk_rope_dim: 4, v_dim: 6 })],
    ["mlp swiglu", around("mlp", { ff: 24 })],
    ["mlp geglu with bias", around("mlp", { ff: 24, act: "geglu", bias: true })],
    ["mlp gelu with bias", around("mlp", { ff: 24, act: "gelu", bias: true })],
    ["mlp relu", around("mlp", { ff: 24, act: "relu" })],
    ["moe with shared experts, gate and router bias", around("moe", { experts: 4, topk: 2, ff: 8, shared: 1, shared_ff: 12, shared_gate: true, router_bias: true })],
    ["rmsnorm", around("rmsnorm", {})],
    ["layernorm without bias", around("layernorm", { bias: false })],
    ["posembed learned", around("posembed", { max: 12 })],
    ["posembed fixed sine", around("posembed", { fixed: true })],
    ["posembed token types", around("posembed", { max: 2, types: true })],
    ["cls_token", around("cls_token", {})],
    ["take_first", around("take_first", {})],
    ["mean_pool", around("mean_pool", {})],
    ["linear on a sequence", around("linear", { out: 5 })],
    ["batchnorm on a sequence", around("batchnorm", {})],
    ["lmhead untied", around("lmhead", { vocab: 20 })],
    ["lmhead tied", around("lmhead", { vocab: 20, tied: true }, "8")],
    ["to_sequence", around("to_sequence", {}, "4x3x5")],
    ["conv2d grouped", around("conv2d", { out: 6, k: 3, groups: 2 }, "4x6x6")],
    ["conv2d depthwise, multiplier 2", around("conv2d", { out: 8, k: 3, s: 2, groups: "depthwise" }, "4x6x6")],
    ["batchnorm on an image", around("batchnorm", {}, "4x6x6")],
    ["layernorm on an image", around("layernorm", {}, "4x6x6")],
    ["linear on an image", around("linear", { out: 3 }, "4x6x6")],
    ["maxpool padded", around("maxpool", { k: 3, s: 2, p: 1 }, "4x6x6")],
    ["avgpool", around("avgpool", { k: 2 }, "4x6x6")],
    ["gap", around("gap", {}, "4x6x6")],
    ["flatten", around("flatten", {}, "4x6x6")],
    ["softmax on an image", around("softmax", {}, "4x6x6")],
  ];
  for (const activation of ["relu", "relu6", "gelu", "silu", "sigmoid", "tanh", "hardswish", "softmax", "dropout", "identity"]) cases.push([activation, around(activation, activation === "dropout" ? { rate: 0.3 } : {})]);

  for (const [name, model] of cases) {
    test(name, () => {
      const graph = buildGraph(model);
      expect(graph.paramCount).toBe(analyze(model).params);
      const shape = tf.tidy(() => graph.apply(feedsFor(graph), { training: true, step: 1 }).shape);
      expect(shape.slice(1)).toEqual(tfShape(analyze(model).outputShape!));
      graph.dispose();
    });
  }

  test("cross_attention reads a memory of a different length", () => {
    const b = new Builder("cross", "language");
    const src = b.input("5", "src");
    const tgt = b.input("7", "tgt");
    const memory = b.node("embedding", src, { vocab: 10, dim: 12 });
    const stream = b.node("embedding", tgt, { vocab: 10, dim: 12, tied: true });
    const c = b.node("cross_attention", [stream, memory], { heads: 4, kv: 2, bias: true });
    b.node("lmhead", c, { vocab: 10, tied: true });
    const model = b.build();
    const graph = buildGraph(model);
    expect(graph.paramCount).toBe(analyze(model).params);
    const shape = tf.tidy(() => graph.apply(feedsFor(graph), { training: false }).shape);
    expect(shape).toEqual([2, 7, 10]);
    graph.dispose();
  });

  test("add, mul (broadcast gate) and concat", () => {
    const b = new Builder("joins", "vision");
    const x = b.input("4x6x6");
    const gate = b.node("sigmoid", b.node("gap", x));
    const gated = b.node("mul", [x, gate]);
    const sum = b.node("add", [gated, x]);
    b.node("concat", [sum, x]);
    const model = b.build();
    const graph = buildGraph(model);
    const shape = tf.tidy(() => graph.apply(feedsFor(graph), { training: false }).shape);
    expect(shape).toEqual([2, 6, 6, 8]);
    graph.dispose();
  });
});

describe("attention semantics", () => {
  // Changing the last token must leave earlier outputs alone in a causal
  // layer, and change them in a bidirectional one.
  const earlierChange = (causal: boolean) => {
    const model = around("attention", { heads: 2, causal, rope: true }, "8");
    const graph = buildGraph(model, { seed: 5 });
    const run = (last: number) =>
      tf.tidy(() => {
        const ids = tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, last]], [1, 8], "int32");
        const id = graph.inputs[0].id;
        return Array.from(graph.apply({ [id]: ids }, { training: false }).slice([0, 0, 0], [1, 7, 16]).dataSync());
      });
    const a = run(8);
    const b = run(9);
    graph.dispose();
    return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  };
  test("causal attention does not look ahead", () => expect(earlierChange(true)).toBeLessThan(1e-6));
  test("bidirectional attention does", () => expect(earlierChange(false)).toBeGreaterThan(1e-4));

  test("a tied output layer and a tied embedding hold no weights of their own and share the table", () => {
    const b = new Builder("tied", "language");
    const x = b.input("4");
    const e = b.node("embedding", x, { vocab: 11, dim: 8 });
    b.node("lmhead", e, { vocab: 11, tied: true });
    const graph = buildGraph(b.build());
    expect(graph.trainable.length).toBe(1);
    expect(graph.paramCount).toBe(88);
    graph.dispose();
  });
});

describe("initialisation", () => {
  const firstValues = (model: Model, seed: number) => {
    const graph = buildGraph(model, { seed });
    const values = Array.from(graph.trainable[1].dataSync().slice(0, 6));
    graph.dispose();
    return values;
  };
  test("the same seed gives the same weights, a different seed different ones", () => {
    const model = prepareForTraining(catalogModel("gpt2")!, { dim: 32, layers: 1 });
    if (!model.ok) throw new Error(model.reason);
    expect(firstValues(model.model, 3)).toEqual(firstValues(model.model, 3));
    expect(firstValues(model.model, 3)).not.toEqual(firstValues(model.model, 4));
  });
  test("repeated blocks get separate weights per copy", () => {
    const model = shrinkLanguageModel(catalogModel("smollm2-135m")!, { dim: 32, layers: 3, vocab: 20, seqLen: 8 });
    const graph = buildGraph(model);
    const queries = graph.trainable.filter((v) => v.name.endsWith("/_q"));
    expect(queries.length).toBe(3);
    expect(Array.from(queries[0].dataSync().slice(0, 4))).not.toEqual(Array.from(queries[1].dataSync().slice(0, 4)));
    // ...and all copies report gradient norms against the one engine node.
    expect(new Set(queries.map((v) => graph.nodeOfVariable.get(v.name))).size).toBe(1);
    graph.dispose();
  });
});
