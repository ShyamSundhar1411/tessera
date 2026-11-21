import * as tf from "@tensorflow/tfjs";
import { asyncBufferFromUrl, parquetMetadataAsync, parquetReadObjects, type AsyncBuffer, type FileMetaData } from "hyparquet";
import type { Shape } from "tessera-engine";

// The three classic image datasets, fetched straight from public hosts that
// allow cross-origin requests, so the static site needs no server. Each loader
// downloads only what it needs for the requested number of examples.
//
// Sources (checked by hand, all send Access-Control-Allow-Origin: *):
// - MNIST: the tf.js examples sprite on Google Cloud Storage, 65,000 digits in
//   one 784-pixel-wide PNG (first 55,000 train, last 10,000 test) plus one-hot
//   labels as raw bytes.
// - Fashion-MNIST: Zalando's original IDX files on GitHub, gzipped. The
//   download is streamed and stopped once enough images have arrived.
// - CIFAR-10: the uoft-cs/cifar10 Parquet files on Hugging Face, one PNG per
//   row. Row groups hold 100 images, so only the byte range covering the rows
//   asked for is downloaded.

export type DatasetName = "mnist" | "fashion-mnist" | "cifar10";

export interface Dataset {
  name: DatasetName | string;
  /** Display name, such as "MNIST". */
  label: string;
  /** One image's shape, channels first like the engine: [1, 28, 28] or [3, 32, 32]. */
  shape: Shape;
  classes: string[];
  /** Images as [N, H, W, C] float32 in 0..1. */
  xTrain: tf.Tensor4D;
  /** One-hot labels [N, classes]. */
  yTrain: tf.Tensor2D;
  xTest: tf.Tensor4D;
  yTest: tf.Tensor2D;
  /** The test labels as class indices, for the confusion matrix. */
  testLabels: Uint8Array;
}

export interface DatasetProgress {
  stage: "download" | "decode";
  /** Bytes or images done so far. */
  loaded: number;
  total?: number;
}

export interface DatasetInfo {
  label: string;
  shape: Shape;
  classes: string[];
  maxTrain: number;
  maxTest: number;
}

export const DATASETS: Record<DatasetName, DatasetInfo> = {
  mnist: { label: "MNIST", shape: [1, 28, 28], classes: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], maxTrain: 55_000, maxTest: 10_000 },
  "fashion-mnist": {
    label: "Fashion-MNIST",
    shape: [1, 28, 28],
    classes: ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    maxTrain: 60_000,
    maxTest: 10_000,
  },
  cifar10: {
    label: "CIFAR-10",
    shape: [3, 32, 32],
    classes: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    maxTrain: 50_000,
    maxTest: 10_000,
  },
};

const MNIST_IMAGES = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";
const FASHION_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/";
const CIFAR_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/";

export async function loadDataset(
  name: DatasetName,
  counts: { train: number; test: number },
  onProgress?: (progress: DatasetProgress) => void,
): Promise<Dataset> {
  const info = DATASETS[name];
  if (!info) throw new Error(`unknown dataset "${name}"`);
  if (counts.train > info.maxTrain || counts.test > info.maxTest)
    throw new Error(`${info.label} has ${info.maxTrain.toLocaleString()} training and ${info.maxTest.toLocaleString()} test images; asked for ${counts.train.toLocaleString()} and ${counts.test.toLocaleString()}`);
  const raw = name === "mnist" ? await loadMnist(counts, onProgress) : name === "fashion-mnist" ? await loadFashion(counts, onProgress) : await loadCifar(counts, onProgress);
  return makeDataset({ name, label: info.label, shape: info.shape, classes: info.classes, ...raw });
}

export interface RawSplits {
  /** Pixels as N*H*W*C float32 in 0..1. */
  trainPixels: Float32Array;
  trainLabels: Uint8Array;
  testPixels: Float32Array;
  testLabels: Uint8Array;
}

/** Wraps raw arrays as tensors. Also handy for tests and for data the user brings. */
export function makeDataset(args: { name: string; label: string; shape: Shape; classes: string[] } & RawSplits): Dataset {
  const [c, h, w] = args.shape;
  const k = args.classes.length;
  const images = (pixels: Float32Array, n: number) => tf.tensor4d(pixels, [n, h, w, c], "float32");
  const oneHot = (labels: Uint8Array) => tf.oneHot(tf.tensor1d(labels, "int32"), k).toFloat() as tf.Tensor2D;
  return {
    name: args.name,
    label: args.label,
    shape: args.shape,
    classes: args.classes,
    xTrain: images(args.trainPixels, args.trainLabels.length),
    yTrain: oneHot(args.trainLabels),
    xTest: images(args.testPixels, args.testLabels.length),
    yTest: oneHot(args.testLabels),
    testLabels: args.testLabels,
  };
}

export function disposeDataset(dataset: Dataset) {
  tf.dispose([dataset.xTrain, dataset.yTrain, dataset.xTest, dataset.yTest]);
}

// -- MNIST ---------------------------------------------------------------------

async function loadMnist(counts: { train: number; test: number }, onProgress?: (p: DatasetProgress) => void): Promise<RawSplits> {
  const [spriteBytes, labelBytes] = await Promise.all([download(MNIST_IMAGES, onProgress), download(MNIST_LABELS)]);
  const sprite = await decodeImage(new Blob([spriteBytes as BlobPart], { type: "image/png" }));
  const pixels = 784;
  // One row of the sprite is one digit; read only the rows needed, in chunks,
  // because a canvas the size of the whole sprite is too big for some browsers.
  const readRows = (start: number, count: number) => {
    const out = new Float32Array(count * pixels);
    const chunk = 5000;
    for (let done = 0; done < count; done += chunk) {
      const rows = Math.min(chunk, count - done);
      const data = drawRegion(sprite, 0, start + done, pixels, rows);
      // The sprite is grey, so the red channel is the value.
      for (let i = 0; i < rows * pixels; i++) out[done * pixels + i] = data[i * 4] / 255;
      onProgress?.({ stage: "decode", loaded: done + rows, total: count });
    }
    return out;
  };
  const labelsFrom = (start: number, count: number) => {
    const out = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      const row = labelBytes.subarray((start + i) * 10, (start + i + 1) * 10);
      out[i] = row.indexOf(1);
    }
    return out;
  };
  const testStart = 55_000;
  return {
    trainPixels: readRows(0, counts.train),
    trainLabels: labelsFrom(0, counts.train),
    testPixels: readRows(testStart, counts.test),
    testLabels: labelsFrom(testStart, counts.test),
  };
}

// -- Fashion-MNIST ---------------------------------------------------------------

async function loadFashion(counts: { train: number; test: number }, onProgress?: (p: DatasetProgress) => void): Promise<RawSplits> {
  // IDX files: a 16-byte header then 784 bytes per image, or an 8-byte header
  // then one byte per label.
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    gunzipPrefix(`${FASHION_BASE}train-images-idx3-ubyte.gz`, 16 + counts.train * 784, onProgress),
    gunzipPrefix(`${FASHION_BASE}train-labels-idx1-ubyte.gz`, 8 + counts.train),
    gunzipPrefix(`${FASHION_BASE}t10k-images-idx3-ubyte.gz`, 16 + counts.test * 784),
    gunzipPrefix(`${FASHION_BASE}t10k-labels-idx1-ubyte.gz`, 8 + counts.test),
  ]);
  const scale = (bytes: Uint8Array) => Float32Array.from(bytes.subarray(16), (v) => v / 255);
  return {
    trainPixels: scale(trainImages),
    trainLabels: trainLabels.slice(8),
    testPixels: scale(testImages),
    testLabels: testLabels.slice(8),
  };
}

/** Streams a gzipped file through the browser's own decompressor and stops once `bytes` have come out. */
async function gunzipPrefix(url: string, bytes: number, onProgress?: (p: DatasetProgress) => void): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`could not download ${url} (${response.status})`);
  const reader = response.body.pipeThrough(new DecompressionStream("gzip")).getReader();
  const out = new Uint8Array(bytes);
  let filled = 0;
  while (filled < bytes) {
    const { done, value } = await reader.read();
    if (done) break;
    const take = Math.min(value.length, bytes - filled);
    out.set(value.subarray(0, take), filled);
    filled += take;
    onProgress?.({ stage: "download", loaded: filled, total: bytes });
  }
  await reader.cancel().catch(() => {});
  if (filled < bytes) throw new Error(`${url} ended after ${filled} of ${bytes} bytes`);
  return out;
}

// -- CIFAR-10 --------------------------------------------------------------------

async function loadCifar(counts: { train: number; test: number }, onProgress?: (p: DatasetProgress) => void): Promise<RawSplits> {
  const [train, test] = await Promise.all([
    readParquetImages(`${CIFAR_BASE}train-00000-of-00001.parquet`, counts.train, onProgress),
    readParquetImages(`${CIFAR_BASE}test-00000-of-00001.parquet`, counts.test),
  ]);
  return { trainPixels: train.pixels, trainLabels: train.labels, testPixels: test.pixels, testLabels: test.labels };
}

async function readParquetImages(url: string, rows: number, onProgress?: (p: DatasetProgress) => void) {
  const remote = await asyncBufferFromUrl({ url });
  const metadata = await parquetMetadataAsync(remote);
  // Fetch the bytes of the first row groups in one request instead of one per group.
  const file = await prefetchRowGroups(remote, metadata, url, rows, onProgress);
  const records = (await parquetReadObjects({ file, metadata, columns: ["img", "label"], rowStart: 0, rowEnd: rows, utf8: false })) as { img: { bytes: Uint8Array }; label: bigint | number }[];
  const pixels = new Float32Array(rows * 32 * 32 * 3);
  const labels = new Uint8Array(rows);
  for (let i = 0; i < records.length; i++) {
    const image = await decodeImage(new Blob([records[i].img.bytes as BlobPart], { type: "image/png" }));
    const data = drawRegion(image, 0, 0, 32, 32);
    for (let p = 0; p < 32 * 32; p++) {
      pixels[(i * 1024 + p) * 3] = data[p * 4] / 255;
      pixels[(i * 1024 + p) * 3 + 1] = data[p * 4 + 1] / 255;
      pixels[(i * 1024 + p) * 3 + 2] = data[p * 4 + 2] / 255;
    }
    if ("close" in image) image.close();
    labels[i] = Number(records[i].label);
    if (i % 250 === 0 || i === records.length - 1) onProgress?.({ stage: "decode", loaded: i + 1, total: rows });
  }
  return { pixels, labels };
}

async function prefetchRowGroups(remote: AsyncBuffer, metadata: FileMetaData, url: string, rows: number, onProgress?: (p: DatasetProgress) => void): Promise<AsyncBuffer> {
  let start = Infinity;
  let end = 0;
  let covered = 0;
  for (const group of metadata.row_groups) {
    if (covered >= rows) break;
    covered += Number(group.num_rows);
    for (const column of group.columns) {
      const meta = column.meta_data;
      if (!meta) continue;
      const begin = Number(meta.dictionary_page_offset ?? meta.data_page_offset);
      start = Math.min(start, begin);
      end = Math.max(end, begin + Number(meta.total_compressed_size));
    }
  }
  if (!Number.isFinite(start)) return remote;
  const block = await download(url, onProgress, { Range: `bytes=${start}-${end - 1}` });
  return {
    byteLength: remote.byteLength,
    slice: (from: number, to = remote.byteLength) =>
      from >= start && to <= end ? block.slice(from - start, to - start).buffer : remote.slice(from, to),
  };
}

// -- Shared helpers ------------------------------------------------------------

/** Fetches a file, reporting bytes as they arrive. */
async function download(url: string, onProgress?: (p: DatasetProgress) => void, headers?: Record<string, string>): Promise<Uint8Array> {
  const response = await fetch(url, { headers });
  if (!response.ok || !response.body) throw new Error(`could not download ${url} (${response.status})`);
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
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

type Drawable = ImageBitmap | HTMLImageElement;

/** Decodes an image with the browser. Falls back to an <img> element where createImageBitmap refuses very tall images. */
async function decodeImage(blob: Blob): Promise<Drawable> {
  try {
    return await createImageBitmap(blob);
  } catch (error) {
    if (typeof document === "undefined") throw error;
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return image;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

/** Draws part of an image onto a canvas and returns its RGBA bytes. Works in a page or a worker. */
function drawRegion(image: Drawable, x: number, y: number, width: number, height: number): Uint8ClampedArray {
  const canvas = makeCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!context) throw new Error("this browser cannot draw images to a canvas");
  context.drawImage(image, x, y, width, height, 0, 0, width, height);
  return context.getImageData(0, 0, width, height).data;
}

export function makeCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}
