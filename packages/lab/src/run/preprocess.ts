import type { InputSpec } from "./registry";

// The pure parts of preparing an image, kept apart from the canvas so they can
// be tested without a browser.

/**
 * The part of the original image that a "resize the short edge, then crop the
 * centre" pipeline keeps. Drawing this rectangle onto a size×size canvas does
 * both steps in one go.
 */
export function cropRect(width: number, height: number, spec: Pick<InputSpec, "size" | "resize">) {
  const scale = spec.resize / Math.min(width, height);
  const side = spec.size / scale;
  return { sx: (width - side) / 2, sy: (height - side) / 2, sw: side, sh: side };
}

/** RGBA bytes to a normalised [1, 3, size, size] float tensor, channels first as ONNX models expect. */
export function toNchw(rgba: ArrayLike<number>, size: number, spec: Pick<InputSpec, "mean" | "std" | "channelOrder">): Float32Array {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  const order = spec.channelOrder === "bgr" ? [2, 1, 0] : [0, 1, 2];
  for (let c = 0; c < 3; c++) {
    const source = order[c];
    const mean = spec.mean[source];
    const std = spec.std[source];
    for (let p = 0; p < plane; p++) out[c * plane + p] = (rgba[p * 4 + source] / 255 - mean) / std;
  }
  return out;
}

export function softmax(logits: ArrayLike<number>): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += out[i] = Math.exp(logits[i] - max);
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function topK(probabilities: ArrayLike<number>, labels: string[], k = 5) {
  return Array.from(probabilities, (probability, index) => ({ index, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k)
    .map(({ index, probability }) => ({ index, label: labels[index] ?? `class ${index}`, probability }));
}
