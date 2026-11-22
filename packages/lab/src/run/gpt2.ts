import * as tf from "@tensorflow/tfjs";
import { catalogModel, type Model } from "tessera-engine";
import { buildGraph, type GraphModel } from "../train/graph";
import { Gpt2Tokenizer } from "./bpe";

// OpenAI's real GPT-2 (124M) running in the same tf.js graph as everything
// else, so the trace view can show what the trained model does at every
// layer. The weights come straight from Hugging Face's safetensors file.

const REPO = "https://huggingface.co/openai-community/gpt2/resolve/main";
export const GPT2_FILES = { weights: `${REPO}/model.safetensors`, vocab: `${REPO}/vocab.json`, merges: `${REPO}/merges.txt` };
export const GPT2_BYTES = 548_105_171;
const CACHE = "tessera-weights-v1";

export interface Gpt2Progress {
  stage: "download" | "weights";
  loaded: number;
  total: number;
}

export interface SafeTensor {
  shape: number[];
  data: Float32Array;
}

/** Reads a safetensors file: an 8-byte header length, a JSON header, then the raw tensors. Only F32 tensors are kept. */
export function parseSafetensors(buffer: ArrayBuffer): Map<string, SafeTensor> {
  const view = new DataView(buffer);
  const length = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const start = 8 + length;
  const out = new Map<string, SafeTensor>();
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    // Integer bookkeeping such as batch norm's num_batches_tracked is not needed to run a model.
    if (entry.dtype !== "F32") continue;
    const [from, to] = entry.data_offsets;
    // Copied, because the offset is not always a multiple of 4.
    out.set(name, { shape: entry.shape, data: new Float32Array(buffer.slice(start + from, start + to)) });
  }
  return out;
}

/** Downloads a file once and keeps it in the browser's cache storage, reporting progress. */
export async function fetchCached(url: string, onProgress?: (loaded: number, total: number) => void): Promise<Response> {
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE).catch(() => null) : null;
  const hit = await cache?.match(url);
  if (hit) return hit;
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`could not download ${url} (${response.status})`);
  const total = Number(response.headers.get("content-length") ?? 0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const blob = new Blob(chunks as BlobPart[]);
  const copy = new Response(blob, { headers: { "content-type": response.headers.get("content-type") ?? "application/octet-stream" } });
  await cache?.put(url, copy.clone()).catch(() => undefined);
  return copy;
}

/** The catalog's GPT-2 with a shorter input, so a trace is quick. Node ids match the weight names below. */
export function gpt2Model(seqLen: number): Model {
  const model = catalogModel("gpt2")!;
  const input = model.nodes.find((node) => node.op === "input")!;
  input.args = { ...input.args, shape: String(seqLen) };
  return { ...model, seqLen };
}

/** Graph variable (by step key and part) for each Hugging Face tensor, splitting the fused query/key/value matrix. */
export function gpt2Assignments(tensors: Map<string, SafeTensor>, layers: number): Map<string, SafeTensor> {
  const get = (name: string) => {
    const tensor = tensors.get(name) ?? tensors.get(`transformer.${name}`);
    if (!tensor) throw new Error(`the weights have no ${name}`);
    return tensor;
  };
  const out = new Map<string, SafeTensor>();
  const norm = (key: string, name: string) => {
    out.set(`${key}/_gamma`, get(`${name}.weight`));
    out.set(`${key}/_beta`, get(`${name}.bias`));
  };
  const dense = (key: string, part: string, name: string) => {
    out.set(`${key}/${part}`, get(`${name}.weight`));
    out.set(`${key}/${part}_bias`, get(`${name}.bias`));
  };
  out.set("embedding/w", get("wte.weight"));
  out.set("posembed/w", get("wpe.weight"));
  for (let i = 0; i < layers; i++) {
    const h = `h.${i}`;
    norm(`layernorm_r${i}`, `${h}.ln_1`);
    norm(`layernorm_2_r${i}`, `${h}.ln_2`);
    // c_attn holds query, key and value side by side: [d, 3d] and [3d].
    const w = get(`${h}.attn.c_attn.weight`);
    const b = get(`${h}.attn.c_attn.bias`);
    const [d, three] = w.shape;
    const width = three / 3;
    ["_q", "_k", "_v"].forEach((part, p) => {
      const data = new Float32Array(d * width);
      for (let r = 0; r < d; r++) data.set(w.data.subarray(r * three + p * width, r * three + (p + 1) * width), r * width);
      out.set(`attention_r${i}/${part}`, { shape: [d, width], data });
      out.set(`attention_r${i}/${part}_bias`, { shape: [width], data: b.data.slice(p * width, (p + 1) * width) });
    });
    dense(`attention_r${i}`, "_o", `${h}.attn.c_proj`);
    dense(`mlp_r${i}`, "_up", `${h}.mlp.c_fc`);
    dense(`mlp_r${i}`, "_down", `${h}.mlp.c_proj`);
  }
  norm("layernorm_3", "ln_f");
  return out;
}

/** Copies weights into a built graph; every variable (trained, and state such as running statistics) must get one, with the same shape. */
export function assignWeights(graph: GraphModel, weights: Map<string, SafeTensor>): void {
  for (const variable of [...graph.trainable, ...graph.state]) {
    // Names are "<build prefix>/<step key>/<part>".
    const key = variable.name.split("/").slice(1).join("/");
    const tensor = weights.get(key);
    if (!tensor) throw new Error(`no weights for ${key}`);
    if (tensor.shape.join("x") !== variable.shape.join("x")) throw new Error(`${key} is ${variable.shape.join("×")} but the weights are ${tensor.shape.join("×")}`);
    const value = tf.tensor(tensor.data, tensor.shape, "float32");
    variable.assign(value);
    value.dispose();
  }
}

export interface Gpt2 {
  graph: GraphModel;
  model: Model;
  tokenizer: Gpt2Tokenizer;
}

/** Downloads (once) and builds the real GPT-2 for sequences of seqLen tokens. */
export async function loadGpt2(seqLen = 64, onProgress?: (progress: Gpt2Progress) => void): Promise<Gpt2> {
  const [vocab, merges] = await Promise.all([fetchCached(GPT2_FILES.vocab).then((r) => r.json() as Promise<Record<string, number>>), fetchCached(GPT2_FILES.merges).then((r) => r.text())]);
  const response = await fetchCached(GPT2_FILES.weights, (loaded, total) => onProgress?.({ stage: "download", loaded, total: total || GPT2_BYTES }));
  const buffer = await response.arrayBuffer();
  onProgress?.({ stage: "weights", loaded: 0, total: 1 });
  const tensors = parseSafetensors(buffer);
  const model = gpt2Model(seqLen);
  const graph = buildGraph(model, { seed: 0 });
  try {
    assignWeights(graph, gpt2Assignments(tensors, 12));
  } catch (error) {
    graph.dispose();
    throw error;
  }
  onProgress?.({ stage: "weights", loaded: 1, total: 1 });
  return { graph, model, tokenizer: new Gpt2Tokenizer(vocab, merges) };
}

/** Scores for the token after `ids` (the last seqLen of them), without keeping every layer as a trace does. */
export function nextTokenLogits(graph: GraphModel, ids: number[]): Float32Array {
  const id = graph.inputs[0].id;
  const length = graph.inputShapes[id][0];
  const window = ids.slice(-length);
  const feed = new Int32Array(length).fill(50256);
  window.forEach((t, i) => (feed[i] = t));
  const row = tf.tidy(() => {
    const logits = graph.apply({ [id]: tf.tensor2d(feed, [1, length], "int32") }, { training: false });
    return tf.slice(logits, [0, window.length - 1, 0], [1, 1, -1]).reshape([-1]);
  });
  const out = row.dataSync() as Float32Array;
  row.dispose();
  return Float32Array.from(out);
}
