import type { Node } from "tessera-engine";

// Reading a traced layer's numbers back as plain words, and showing how any
// one of its numbers was computed from the layer's inputs and weights. Works
// on tf's layouts: images [H, W, C], sequences [S, D], vectors [N].

export interface LayerValues {
  node: Node;
  shape: number[];
  data: Float32Array;
  attention?: { shape: number[]; data: Float32Array };
  rows?: string[];
  keys?: string[];
}

type Weights = Record<string, { shape: number[]; data: Float32Array }>;

const pct = (v: number) => `${Math.round(v * 100)}%`;
const num = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const quote = (s: string | undefined) => `"${s ?? "?"}"`;

function norm(data: Float32Array, start = 0, end = data.length) {
  let s = 0;
  for (let i = start; i < end; i++) s += data[i] * data[i];
  return Math.sqrt(s);
}

function moments(data: Float32Array, start = 0, end = data.length) {
  let sum = 0;
  for (let i = start; i < end; i++) sum += data[i];
  const mean = sum / (end - start);
  let sq = 0;
  for (let i = start; i < end; i++) sq += (data[i] - mean) ** 2;
  return { mean, std: Math.sqrt(sq / (end - start)) };
}

/** Where on a map a position sits, in words. */
function place(r: number, c: number, h: number, w: number) {
  const v = r < h / 3 ? "top" : r >= (2 * h) / 3 ? "bottom" : "middle";
  const s = c < w / 3 ? "left" : c >= (2 * w) / 3 ? "right" : "centre";
  return v === "middle" && s === "centre" ? "the centre" : `the ${v} ${s}`;
}

/** The row a sequence layer is about: the picked token, kept inside the layer's length. */
const rowOf = (layer: LayerValues, token: number) => Math.max(0, Math.min(token, layer.shape[0] - 1));

/**
 * What a layer did, in a sentence, from its actual numbers. `inputs` are the
 * layers it read; `token` is the picked position for sequences.
 */
export function insight(layer: LayerValues, inputs: LayerValues[], token: number): string | null {
  const { node, shape, data } = layer;
  const x = inputs[0];
  const seq = shape.length === 2;
  const r = seq ? rowOf(layer, token) : 0;
  const d = seq ? shape[1] : 0;
  const name = seq ? quote(layer.rows?.[r]) : "";
  const span = (l: LayerValues) => (seq && l.shape.length === 2 ? [r * l.shape[1], (r + 1) * l.shape[1]] : [0, l.data.length]);
  switch (node.op) {
    case "input":
      return shape.length === 3 ? `A ${shape[1]}×${shape[0]} picture with ${shape[2]} colour channel${shape[2] > 1 ? "s" : ""}: ${data.length.toLocaleString("en-US")} numbers in all.` : null;
    case "embedding": {
      if (!seq) return null;
      // Which other token in the text has the most similar vector.
      const v = data.subarray(r * d, (r + 1) * d);
      let best = -1;
      let bestCos = -2;
      const used = layer.rows?.findLastIndex((label) => label !== "·") ?? shape[0] - 1;
      for (let o = 0; o <= used; o++) {
        if (o === r || layer.rows?.[o] === layer.rows?.[r]) continue;
        const u = data.subarray(o * d, (o + 1) * d);
        let dot = 0;
        for (let i = 0; i < d; i++) dot += u[i] * v[i];
        const cos = dot / (norm(u) * norm(v) || 1);
        if (cos > bestCos) [best, bestCos] = [o, cos];
      }
      return `${name} becomes a list of ${d} numbers.${best >= 0 ? ` Of the other tokens here, ${quote(layer.rows?.[best])} has the most similar list (cosine ${bestCos.toFixed(2)}).` : ""}`;
    }
    case "posembed": {
      if (!x || !seq) return null;
      const [a, b] = span(x);
      let change = 0;
      for (let i = a; i < b; i++) change += (data[i] - x.data[i]) ** 2;
      return `Position ${r + 1} is mixed into ${name}: its vector moved by ${pct(Math.sqrt(change) / (norm(x.data, a, b) || 1))} of its length, the same shift every token at that position gets.`;
    }
    case "layernorm":
    case "rmsnorm":
    case "batchnorm": {
      if (!x) return null;
      const [a, b] = span(x);
      const before = moments(x.data, a, b);
      const after = moments(data, a, b);
      return `${seq ? `For ${name}, the` : "The"} numbers went from mean ${num(before.mean)} and spread ${num(before.std)} to mean ${num(after.mean)} and spread ${num(after.std)}, so the next layer sees a steady scale.`;
    }
    case "relu":
    case "relu6": {
      let zeros = 0;
      for (let i = 0; i < data.length; i++) if (data[i] === 0) zeros++;
      return `Set ${pct(zeros / data.length)} of the numbers to zero (every negative one) and kept the rest. Those zeros are why the maps above go dark in patches.`;
    }
    case "gelu":
    case "silu":
    case "hardswish": {
      let small = 0;
      for (let i = 0; i < data.length; i++) if (Math.abs(data[i]) < 0.05) small++;
      return `Pushed ${pct(small / data.length)} of the numbers to almost zero, softly: small negatives leak through a little, positives pass.`;
    }
    case "sigmoid":
    case "tanh": {
      const { mean } = moments(data);
      return `Squashed every number into ${node.op === "sigmoid" ? "0 to 1" : "-1 to 1"}; the average is now ${num(mean)}.`;
    }
    case "dropout":
      return "Does nothing here: dropout only switches numbers off while training. The values pass through unchanged.";
    case "attention":
    case "cross_attention":
    case "mla": {
      const att = layer.attention;
      if (!att) return null;
      const [heads, q, k] = att.shape;
      const row = Math.min(r, q - 1);
      const mean = new Float32Array(k);
      for (let h = 0; h < heads; h++) for (let j = 0; j < k; j++) mean[j] += att.data[(h * q + row) * k + j] / heads;
      const order = Array.from(mean.keys()).sort((a, b) => mean[b] - mean[a]);
      const keys = layer.keys ?? [];
      return `${quote(layer.rows?.[row])} ${node.op === "cross_attention" ? "reads the source and " : ""}looks most at ${quote(keys[order[0]])} (${pct(mean[order[0]])} on average over ${heads} heads)${order[1] !== undefined ? `, then ${quote(keys[order[1]])} (${pct(mean[order[1]])})` : ""}.`;
    }
    case "add": {
      if (inputs.length < 2) return null;
      const [a, b] = span(inputs[0]);
      const stream = norm(inputs[1].data, a, b);
      const change = norm(inputs[0].data, a, b);
      return `${seq ? `For ${name}, the` : "The"} block's output is added back onto what came in: a change ${pct(change / (stream || 1))} the size of the stream. ${change < stream ? "Small changes like this are how deep networks refine rather than overwrite." : "The block rewrote most of it."}`;
    }
    case "mul":
      return inputs[1] ? `Each channel was scaled by its gate; the gates average ${num(moments(inputs[1].data).mean)}.` : null;
    case "concat":
      return `Laid ${inputs.length} inputs side by side: ${inputs.map((l) => l.shape[l.shape.length - 1]).join(" + ")} = ${shape[shape.length - 1]} channels, nothing lost.`;
    case "conv2d":
    case "upconv2d": {
      if (shape.length !== 3) return null;
      const [h, w, c] = shape;
      let bestK = 0;
      let bestMean = -Infinity;
      for (let k = 0; k < c; k++) {
        let s = 0;
        for (let p = 0; p < h * w; p++) s += data[p * c + k];
        if (s / (h * w) > bestMean) [bestK, bestMean] = [k, s / (h * w)];
      }
      let peak = 0;
      for (let p = 0; p < h * w; p++) if (data[p * c + bestK] > data[peak * c + bestK]) peak = p;
      return `Filter #${bestK + 1} of ${c} responds most strongly overall, peaking at ${place(Math.floor(peak / w), peak % w, h, w)} of the map.`;
    }
    case "maxpool":
    case "avgpool":
      return x ? `Each window became one number, its ${node.op === "maxpool" ? "largest" : "average"}: ${x.shape[0]}×${x.shape[1]} shrank to ${shape[0]}×${shape[1]}, keeping ${pct((shape[0] * shape[1]) / (x.shape[0] * x.shape[1]))} of the positions.` : null;
    case "gap": {
      let best = 0;
      for (let k = 1; k < data.length; k++) if (data[k] > data[best]) best = k;
      return `Each channel became its average over the picture. Channel #${best + 1} is the strongest, at ${num(data[best])}.`;
    }
    case "flatten":
      return `The same ${data.length.toLocaleString("en-US")} numbers, laid out as one list.`;
    case "mlp":
    case "moe": {
      if (!x || !seq) return null;
      const [a, b] = span(x);
      return `For ${name}, the feed-forward produced a change ${pct(norm(data, a, b) / (norm(x.data, a, b) || 1))} the size of its input, which the next skip join adds on.`;
    }
    case "linear": {
      if (shape.length !== 1) return null;
      let best = 0;
      for (let k = 1; k < data.length; k++) if (data[k] > data[best]) best = k;
      return `Output #${best + 1} of ${data.length} is the largest, at ${num(data[best])}.`;
    }
    case "cls_token":
      return "A learned summary token now sits in front; by the last layer it has gathered the whole input.";
    case "take_first":
      return "Kept only the summary token's numbers for the classifier.";
    case "to_sequence":
      return `The ${shape[0]} grid positions are now ${shape[0]} tokens.`;
    default:
      return null;
  }
}

export interface Term {
  label: string;
  value: number;
}

export interface Working {
  /** What is being computed, such as 'output 12 for "be"'. */
  target: string;
  /** The arithmetic, top terms first. */
  lines: { text: string; value?: number }[];
  result: number;
}

/**
 * How one number of a layer was made, with the real values. `index` is the
 * flat position in the layer's data. Weights are fetched only for the ops that
 * need them.
 */
export function working(layer: LayerValues, inputs: LayerValues[], index: number, getWeights: () => Weights): Working | null {
  const { node, shape, data } = layer;
  const value = data[index];
  const x = inputs[0];
  const seq = shape.length === 2;
  const d = shape[shape.length - 1];
  const row = seq ? Math.floor(index / d) : 0;
  const col = index % d;
  const where = seq ? `number ${col + 1} for ${quote(layer.rows?.[row])}` : shape.length === 3 ? `channel ${col + 1} at row ${Math.floor(index / d / shape[1]) + 1}, column ${(Math.floor(index / d) % shape[1]) + 1}` : `output ${index + 1}`;
  const top = (terms: Term[], n = 6) => {
    const sorted = [...terms].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const shown = sorted.slice(0, n);
    const rest = sorted.slice(n).reduce((s, t) => s + t.value, 0);
    return [...shown.map((t) => ({ text: t.label, value: t.value })), ...(sorted.length > n ? [{ text: `+ the other ${sorted.length - n} terms`, value: rest }] : [])];
  };
  switch (node.op) {
    case "relu":
    case "relu6":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
    case "hardswish":
      return x ? { target: where, lines: [{ text: `${node.op}(${num(x.data[index])})` }], result: value } : null;
    case "dropout":
    case "flatten":
      return x ? { target: where, lines: [{ text: `copied unchanged: ${num(x.data[index])}` }], result: value } : null;
    case "add":
      return inputs.length >= 2 ? { target: where, lines: inputs.map((l, i) => ({ text: i === 0 ? "from the block" : "skipped past it", value: l.data[index] })), result: value } : null;
    case "mul":
      return inputs.length >= 2 ? { target: where, lines: [{ text: `${num(inputs[0].data[index])} × gate ${num(inputs[1].data[col])}` }], result: value } : null;
    case "embedding":
      return { target: where, lines: [{ text: `row for ${quote(layer.rows?.[row])} in the table, column ${col + 1}: read straight out of the learned table` }], result: value };
    case "posembed":
      return x ? { target: where, lines: [{ text: "token's number", value: x.data[index] }, { text: `position ${row + 1}'s number`, value: value - x.data[index] }], result: value } : null;
    case "layernorm":
    case "rmsnorm": {
      if (!x) return null;
      const w = getWeights();
      const start = seq ? row * d : 0;
      const { mean, std } = moments(x.data, start, start + d);
      const g = w._gamma?.data[col] ?? 1;
      const b = w._beta?.data[col] ?? 0;
      if (node.op === "rmsnorm") {
        const rms = norm(x.data, start, start + d) / Math.sqrt(d);
        return { target: where, lines: [{ text: `x = ${num(x.data[index])}, root mean square of the ${d} numbers = ${num(rms)}` }, { text: `(${num(x.data[index])} ÷ ${num(rms)}) × γ ${num(g)}` }], result: value };
      }
      return { target: where, lines: [{ text: `x = ${num(x.data[index])}, mean μ = ${num(mean)}, spread σ = ${num(std)} over ${d} numbers` }, { text: `(${num(x.data[index])} − ${num(mean)}) ÷ ${num(std)} × γ ${num(g)} + β ${num(b)}` }], result: value };
    }
    case "batchnorm": {
      if (!x) return null;
      const w = getWeights();
      const c = col;
      const m = w.running_mean?.data[c] ?? 0;
      const v = w.running_var?.data[c] ?? 1;
      return { target: where, lines: [{ text: `x = ${num(x.data[index])}; channel ${c + 1} learned mean ${num(m)}, variance ${num(v)}` }, { text: `(${num(x.data[index])} − ${num(m)}) ÷ √${num(v)} × γ ${num(w._gamma?.data[c] ?? 1)} + β ${num(w._beta?.data[c] ?? 0)}` }], result: value };
    }
    case "linear": {
      if (!x) return null;
      const w = getWeights();
      const matrix = w[""];
      if (!matrix) return null;
      const [inputs_, outputs] = matrix.shape;
      const start = seq ? row * inputs_ : 0;
      const terms: Term[] = [];
      for (let i = 0; i < inputs_; i++) terms.push({ label: `input ${i + 1} (${num(x.data[start + i])}) × weight ${num(matrix.data[i * outputs + col])}`, value: x.data[start + i] * matrix.data[i * outputs + col] });
      const bias = w._bias?.data[col];
      return { target: where, lines: [...top(terms), ...(bias !== undefined ? [{ text: "bias", value: bias }] : [])], result: value };
    }
    case "conv2d": {
      if (!x || shape.length !== 3 || x.shape.length !== 3) return null;
      const w = getWeights();
      const filter = w[""];
      if (!filter || filter.shape[3] !== shape[2] || filter.shape[2] !== x.shape[2]) return { target: where, lines: [{ text: "A grouped or depthwise convolution: each output reads only its own channels." }], result: value };
      const [k, , cin, cout] = filter.shape;
      const [, outW] = shape;
      const [inH, inW] = x.shape;
      const s = Number(node.args.s ?? 1);
      const p = node.args.p === undefined ? Math.floor(k / 2) : Number(node.args.p);
      const pos = Math.floor(index / cout);
      const [r, c] = [Math.floor(pos / outW), pos % outW];
      const terms: Term[] = [];
      for (let u = 0; u < k; u++)
        for (let v = 0; v < k; v++) {
          const [y, xx] = [r * s + u - p, c * s + v - p];
          if (y < 0 || xx < 0 || y >= inH || xx >= inW) continue;
          for (let ci = 0; ci < cin; ci++) {
            const input = x.data[(y * inW + xx) * cin + ci];
            const weight = filter.data[((u * k + v) * cin + ci) * cout + col];
            terms.push({ label: `pixel (${y + 1}, ${xx + 1}) channel ${ci + 1}: ${num(input)} × ${num(weight)}`, value: input * weight });
          }
        }
      const bias = w._bias?.data[col];
      return { target: `filter ${col + 1} at row ${r + 1}, column ${c + 1}`, lines: [...top(terms), ...(bias !== undefined ? [{ text: "bias", value: bias }] : [])], result: value };
    }
    case "maxpool":
    case "avgpool": {
      if (!x || shape.length !== 3) return null;
      const [, outW, c] = shape;
      const [inH, inW] = x.shape;
      const k = Number(node.args.k ?? 2);
      const s = Number(node.args.s ?? k);
      const p = Number(node.args.p ?? 0);
      const pos = Math.floor(index / c);
      const [r, cc] = [Math.floor(pos / outW), pos % outW];
      const values: number[] = [];
      for (let u = 0; u < k; u++)
        for (let v = 0; v < k; v++) {
          const [y, xx] = [r * s + u - p, cc * s + v - p];
          if (y >= 0 && xx >= 0 && y < inH && xx < inW) values.push(x.data[(y * inW + xx) * c + col]);
        }
      return { target: where, lines: [{ text: `${node.op === "maxpool" ? "largest" : "average"} of the ${k}×${k} window: ${values.map(num).join(", ")}` }], result: value };
    }
    default:
      return null;
  }
}
