import type { InferenceSession } from "onnxruntime-web";
import { cropRect, softmax, toNchw, topK } from "./preprocess";
import { labelsFor, type PretrainedEntry } from "./registry";

// Runs a pretrained classifier in the browser with ONNX Runtime Web. The
// runtime is imported on first use, so pages that never classify (and the
// static export's build step) do not load it.

export type ClassifiableImage = ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas;

export interface ClassifyProgress {
  stage: "download" | "load" | "run";
  /** Bytes downloaded so far, during "download". */
  loaded?: number;
  total?: number;
}

export interface Prediction {
  index: number;
  label: string;
  probability: number;
}

export interface ClassifyResult {
  top5: Prediction[];
  /** Exactly what the model saw: [1, 3, size, size], normalised. */
  input: { data: Float32Array; dims: [1, 3, number, number] };
  /** The execution provider that ran it: "webgpu" or "wasm". */
  backend: string;
  seconds: number;
}

type Ort = typeof import("onnxruntime-web");

let ortPromise: Promise<Ort> | undefined;
const sessions = new Map<string, Promise<{ session: InferenceSession; backend: string }>>();
const CACHE_NAME = "tessera-models";

async function loadOrt(): Promise<Ort> {
  ortPromise ??= import("onnxruntime-web/webgpu").then((ort) => {
    // Fetch the WebAssembly files from the CDN for exactly this version, so a
    // static export needs no copies of them in public/.
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
    // Threads need a cross-origin isolated page (COOP/COEP headers), which a
    // plain static host does not send.
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    return ort as unknown as Ort;
  });
  return ortPromise;
}

const hasWebGpu = () => typeof navigator !== "undefined" && "gpu" in navigator;

/** Downloads (or loads from the cache) and starts a model once; later calls reuse the session. */
export function loadSession(entry: PretrainedEntry, onProgress?: (progress: ClassifyProgress) => void) {
  let pending = sessions.get(entry.id);
  if (!pending) {
    pending = (async () => {
      const ort = await loadOrt();
      const bytes = await fetchModel(entry.url, onProgress);
      onProgress?.({ stage: "load" });
      const providers = hasWebGpu() ? ["webgpu", "wasm"] : ["wasm"];
      try {
        const session = await ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: "all" });
        return { session, backend: providers[0] };
      } catch (error) {
        if (providers.length === 1) throw error;
        // WebGPU can be present but refuse a model (missing op, adapter lost); WebAssembly always works.
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
        return { session, backend: "wasm" };
      }
    })();
    sessions.set(entry.id, pending);
    // Let a failed download be retried.
    pending.catch(() => sessions.delete(entry.id));
  }
  return pending;
}

export async function releaseSession(entry: PretrainedEntry) {
  const pending = sessions.get(entry.id);
  sessions.delete(entry.id);
  if (pending) await (await pending).session.release().catch(() => {});
}

export async function classify(entry: PretrainedEntry, image: ClassifiableImage, onProgress?: (progress: ClassifyProgress) => void): Promise<ClassifyResult> {
  const { session, backend } = await loadSession(entry, onProgress);
  const ort = await loadOrt();
  onProgress?.({ stage: "run" });
  const size = entry.input.size;
  const data = toNchw(drawCropped(image, entry), size, entry.input);
  const dims: [1, 3, number, number] = [1, 3, size, size];
  const started = performance.now();
  const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", data, dims) });
  const seconds = (performance.now() - started) / 1000;
  const logits = outputs[session.outputNames[0]].data as Float32Array;
  return { top5: topK(softmax(logits), labelsFor(entry), 5), input: { data, dims }, backend, seconds };
}

/** Resizes and centre-crops with the browser's own (high quality) scaling, returning RGBA bytes. */
function drawCropped(image: ClassifiableImage, entry: PretrainedEntry): Uint8ClampedArray {
  const width = "naturalWidth" in image ? image.naturalWidth : image.width;
  const height = "naturalHeight" in image ? image.naturalHeight : image.height;
  if (!width || !height) throw new Error("the image has not loaded yet");
  const size = entry.input.size;
  const { sx, sy, sw, sh } = cropRect(width, height, entry.input);
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(size, size) : Object.assign(document.createElement("canvas"), { width: size, height: size });
  const context = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!context) throw new Error("this browser cannot draw images to a canvas");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, sx, sy, sw, sh, 0, 0, size, size);
  return context.getImageData(0, 0, size, size).data;
}

/** Downloads a model with progress, keeping a copy in the Cache API so the next visit skips the download. */
async function fetchModel(url: string, onProgress?: (progress: ClassifyProgress) => void): Promise<Uint8Array> {
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE_NAME).catch(() => undefined) : undefined;
  const cached = await cache?.match(url);
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    onProgress?.({ stage: "download", loaded: bytes.length, total: bytes.length });
    return bytes;
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`could not download the model (${response.status})`);
  const total = Number(response.headers.get("content-length")) || undefined;
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onProgress?.({ stage: "download", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  // Storage can be full or blocked; the model still runs this time.
  await cache?.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return bytes;
}
