"use client";

import { Pause, Play, RotateCcw, SkipBack, SkipForward, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { OPS, type Model, type Node, type Shape } from "tessera-engine";
import type { Gpt2Tokenizer, GraphModel, PlainFeed, PreparedModel, TaskSpec, TracedLayer } from "tessera-lab";
import { CATEGORY_COLOURS } from "@/lib/display";
import { explain } from "@/lib/equations";
import { useLab, type SlotIndex } from "@/lib/store";
import { trainedFor, useTrained } from "@/lib/trained";
import { cn } from "@/lib/utils";
import Tex from "./tex";
import { RealWeightsButton, WeightsBadge } from "./weights";

// Follow one input through every layer: what each layer computes (its
// equation, with this model's real sizes), and the actual numbers it produced.
// Runs the small version of the model the Train tab trains, with trained
// weights once it has been trained there and fresh random ones before; GPT-2
// can run full size with OpenAI's published weights.

type Lab = typeof import("tessera-lab");

const SLOT_COLOURS = ["#5fb4ff", "#f0569f"];
const DEFAULT_TEXT = "To be, or not to be, that is the question:";
const DEFAULT_SYMBOLS = "3a7f019c42e8";
// The pad's own resolution: enough for the 224×224 ImageNet models.
const PAD = 224;

// -- Shared input --------------------------------------------------------------------

export interface Inputs {
  text: string;
  symbols: string;
  /** The drawing pad; read at whatever size each model wants. */
  pad: HTMLCanvasElement | null;
  /** Bumps when the drawing changes. */
  padVersion: number;
}

function drawGlyph(canvas: HTMLCanvasElement, glyph: string) {
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, PAD, PAD);
  ctx.fillStyle = "#fff";
  ctx.font = `600 ${PAD * 0.72}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(glyph, PAD / 2, PAD * 0.54);
}

/** Pixels of the pad at the model's size, channels last, 0 to 1. */
function readPad(pad: HTMLCanvasElement, h: number, w: number, channels: number): Float32Array {
  const small = document.createElement("canvas");
  small.width = w;
  small.height = h;
  const ctx = small.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(pad, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(h * w * channels);
  for (let p = 0; p < h * w; p++) {
    if (channels === 1) out[p] = (rgba[p * 4] + rgba[p * 4 + 1] + rgba[p * 4 + 2]) / 765;
    else for (let c = 0; c < channels; c++) out[p * channels + c] = rgba[p * 4 + Math.min(c, 2)] / 255;
  }
  return out;
}

/** Ids and labels for every input of a prepared model, from the shared input. */
export function makeFeeds(lab: Lab, prepared: PreparedModel, graph: GraphModel, inputs: Inputs, tokenizer?: Gpt2Tokenizer, normalize?: { mean: number[]; std: number[] }) {
  const { task } = prepared;
  const feeds: Record<string, PlainFeed> = {};
  const labels: Record<string, string[]> = {};
  let used = 0;
  for (const [id, role] of Object.entries(prepared.roles)) {
    const shape = graph.inputShapes[id];
    if (role === "image") {
      const [c, h, w] = shape;
      if (inputs.pad) {
        const data = readPad(inputs.pad, h, w, c);
        if (normalize) for (let i = 0; i < data.length; i++) data[i] = (data[i] - normalize.mean[i % c]) / normalize.std[i % c];
        feeds[id] = { data, shape: [h, w, c] };
      }
      continue;
    }
    const length = shape[0];
    const ids = new Int32Array(length);
    if (role === "zeros") {
      feeds[id] = { data: ids, shape: [length] };
      continue;
    }
    if (tokenizer) {
      // A real model's own tokens; the rest is filled with end-of-text, which a causal model never looks back at.
      const tokens = tokenizer.encode(inputs.text).slice(-length);
      used = Math.max(1, tokens.length);
      ids.fill(50256);
      tokens.forEach((t, i) => (ids[i] = t));
      labels[id] = Array.from(ids, (t, i) => (i < tokens.length ? showToken(tokenizer.decode([t])) : "·"));
    } else if (task.kind === "language-modelling" || task.kind === "masked-tokens") {
      const text = inputs.text.slice(0, length);
      used = Math.max(1, text.length);
      const encoded = lab.encodeChars(text.padEnd(length, " "));
      for (let i = 0; i < length; i++) ids[i] = task.kind === "masked-tokens" && text[i] === "_" ? lab.CHAR_MASK : encoded[i];
      labels[id] = Array.from(text.padEnd(length, " "), (ch) => (ch === " " ? "␣" : ch === "\n" ? "↵" : ch));
    } else {
      const symbols = Array.from(inputs.symbols.toLowerCase().replace(/[^0-9a-f]/g, "").padEnd(length, "0").slice(0, length), (ch) => lab.SYMBOLS.indexOf(ch));
      used = length;
      if (role === "tokens") {
        symbols.forEach((s, i) => (ids[i] = s));
        labels[id] = symbols.map((s) => lab.SYMBOLS[s]);
      } else {
        // seq2seq: the source is symbols 1..16; the decoder reads the expected answer shifted right after a start token.
        const answer = task.seq2seq === "sort" ? [...symbols].sort((a, b) => a - b) : [...symbols].reverse();
        const row = role === "source" ? symbols.map((s) => s + 1) : [0, ...answer.slice(0, -1).map((s) => s + 1)];
        row.forEach((s, i) => (ids[i] = s));
        labels[id] = row.map((s) => (s === 0 ? "▸" : lab.SYMBOLS[s - 1]));
      }
    }
    feeds[id] = { data: ids, shape: [length] };
  }
  return { feeds, labels, used };
}

/** A token as text, with its leading space and line breaks made visible. */
export const showToken = (text: string) => text.replace(/ /g, "␣").replace(/\n/g, "↵");

// -- Running a model -----------------------------------------------------------------

export interface Prepared {
  graph: GraphModel;
  prepared: PreparedModel;
  weights: "random" | "trained" | "pretrained";
  tokenizer?: Gpt2Tokenizer;
  normalize?: { mean: number[]; std: number[] };
  labels?: string[];
}

export function usePreparedModel(slot: SlotIndex, model: Model | undefined) {
  const trainedSlots = useTrained((state) => state.slots);
  const trained = trainedFor(slot, model, trainedSlots);
  const [state, setState] = useState<{ value: Prepared | null; error: string | null; model?: Model }>({ value: null, error: null });

  // Fresh weights on the small version of the model, until the Train tab hands over trained ones.
  useEffect(() => {
    if (!model || trained) return;
    let live = true;
    let graph: GraphModel | null = null;
    void import("tessera-lab").then((lab) => {
      if (!live) return;
      const prepared = lab.prepareForTraining(model);
      if (!prepared.ok) {
        setState({ value: null, error: prepared.reason, model });
        return;
      }
      try {
        graph = lab.buildGraph(prepared.model, { seed: 0 });
        setState({ value: { graph, prepared, weights: "random" }, error: null, model });
      } catch (caught) {
        setState({ value: null, error: caught instanceof Error ? caught.message : String(caught), model });
      }
    });
    return () => {
      live = false;
      graph?.dispose();
    };
  }, [model, trained]);

  const fromTraining = useMemo(() => (trained ? { value: { graph: trained.graph, prepared: trained.prepared, weights: trained.kind, tokenizer: trained.tokenizer, normalize: trained.normalize, labels: trained.labels } as Prepared, error: null } : null), [trained]);
  if (fromTraining) return fromTraining;
  return state.model === model ? state : { value: null, error: null };
}

interface Layer extends TracedLayer {
  node: Node;
  inputShapes: Shape[];
  outputShape: Shape;
  /** Token labels for the rows of a sequence, and for attention's keys. */
  rows?: string[];
  keys?: string[];
  /** The block this copy belongs to, such as "Layer 2 of 4". */
  block?: string;
  /** Kept for activations, so their effect shows as a before and after. */
  before?: Float32Array;
  /** Pixels to show for a normalised input. */
  picture?: Float32Array;
}

interface Trace {
  layers: Layer[];
  ms: number;
  used: number;
  task: TaskSpec;
  classes?: string[];
  /** Text for an output id, for real tokenizers. */
  decode?: (id: number) => string;
}

/** Which input nodes each node depends on, so a sequence's rows get the right token labels. */
function sourcesOf(model: Model) {
  const byId = new Map(model.nodes.map((node) => [node.id, node]));
  const memo = new Map<string, Set<string>>();
  const visit = (id: string): Set<string> => {
    const known = memo.get(id);
    if (known) return known;
    const node = byId.get(id)!;
    const out = new Set<string>();
    memo.set(id, out);
    if (node.op === "input") out.add(id);
    for (const input of node.inputs) for (const s of visit(input)) out.add(s);
    return out;
  };
  return visit;
}

function fitLabels(labels: string[] | undefined, length: number) {
  if (!labels) return Array.from({ length }, (_, i) => String(i));
  if (labels.length === length) return labels;
  if (labels.length + 1 === length) return ["CLS", ...labels];
  return Array.from({ length }, (_, i) => String(i));
}

export function runTrace(lab: Lab, value: Prepared, inputs: Inputs): Trace | null {
  const { graph, prepared, tokenizer, normalize } = value;
  const { feeds, labels, used } = makeFeeds(lab, prepared, graph, inputs, tokenizer, normalize);
  if (Object.keys(feeds).length < graph.inputs.length) return null;
  const started = performance.now();
  const traced = lab.traceArrays(graph, feeds);
  const ms = performance.now() - started;
  const model = prepared.model;
  const byId = new Map(model.nodes.map((node) => [node.id, node]));
  const groups = new Map(model.groups.map((group) => [group.id, group]));
  const sources = sourcesOf(model);
  const info = graph.analysis.info;
  const roleLabels = (id: string) => {
    const from = [...sources(id)];
    // A decoder reads both; its rows are the target's tokens.
    const pick = from.find((s) => prepared.roles[s] === "target") ?? from.find((s) => labels[s]);
    return pick ? labels[pick] : undefined;
  };
  const layers: Layer[] = traced.map((step, index) => {
    const node = byId.get(step.nodeId)!;
    const group = node.group ? groups.get(node.group) : undefined;
    let outer = group;
    while (outer?.parent && !(outer.repeat && outer.repeat > 1)) outer = groups.get(outer.parent);
    const block = outer?.repeat && outer.repeat > 1 ? `${outer.label} ${step.copy + 1} of ${outer.repeat}` : group?.label;
    const sequence = step.shape.length === 2;
    const layer: Layer = {
      ...step,
      node,
      inputShapes: node.inputs.map((id) => info.get(id)!.shape),
      outputShape: info.get(node.id)!.shape,
      block,
      rows: sequence ? fitLabels(roleLabels(node.id), step.shape[0]) : undefined,
    };
    if (step.attention) layer.keys = fitLabels(node.op === "cross_attention" ? roleLabels(node.inputs[1]) : layer.rows, step.attention.shape[2]);
    const previous = traced[index - 1];
    if (OPS[node.op]?.category === "activation" && previous && node.inputs[0] === previous.nodeId) layer.before = previous.data;
    return layer;
  });
  // The input picture as pixels, before any normalisation, for showing.
  const input = layers.find((layer) => layer.node.op === "input" && layer.shape.length === 3);
  if (input && normalize && inputs.pad) input.picture = readPad(inputs.pad, input.shape[0], input.shape[1], input.shape[2]);
  const { task } = prepared;
  const classes =
    value.labels ?? (task.kind === "image-classification" ? lab.DATASETS[task.imageDataset ?? "mnist"].classes : task.kind === "sequence-classification" ? Array.from({ length: task.classes }, (_, i) => `most ${lab.SYMBOLS[i]}`) : undefined);
  return { layers, ms, used, task: prepared.task, classes, decode: tokenizer ? (id: number) => showToken(tokenizer.decode([id])) : undefined };
}

// -- Pictures of numbers -------------------------------------------------------------

const POS = [240, 86, 159];
const NEG = [95, 180, 255];

/** Colours a value: pink for positive, blue for negative, see-through near zero, so it reads on light and dark. */
function paint(img: Uint8ClampedArray, offset: number, v: number, scale: number, sequential: boolean) {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)));
  const colour = sequential || t >= 0 ? POS : NEG;
  img[offset] = colour[0];
  img[offset + 1] = colour[1];
  img[offset + 2] = colour[2];
  img[offset + 3] = Math.round(Math.pow(Math.abs(t), sequential ? 0.6 : 0.8) * 255);
}

function absMax(data: Float32Array, start = 0, end = data.length, stride = 1) {
  let m = 0;
  for (let i = start; i < end; i += stride) m = Math.max(m, Math.abs(data[i]));
  return m;
}

interface HeatmapProps {
  rows: number;
  cols: number;
  at: (r: number, c: number) => number;
  scale: number;
  sequential?: boolean;
  rowLabels?: string[];
  colLabels?: string[];
  describe?: (r: number, c: number, v: number) => string;
  className?: string;
  /** Height per row in CSS pixels; the width always fills. */
  rowHeight?: number;
}

function Heatmap({ rows, cols, at, scale, sequential, rowLabels, colLabels, describe, className, rowHeight }: HeatmapProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d")!;
    const image = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) paint(image.data, (r * cols + c) * 4, at(r, c), scale, !!sequential);
    ctx.putImageData(image, 0, 0);
  }, [rows, cols, at, scale, sequential]);
  const onMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const c = Math.floor(((event.clientX - box.left) / box.width) * cols);
    const r = Math.floor(((event.clientY - box.top) / box.height) * rows);
    if (r >= 0 && r < rows && c >= 0 && c < cols) setHover({ r, c, x: event.clientX - box.left, y: event.clientY - box.top });
  };
  return (
    <div className={cn("relative", className)}>
      <div className="flex">
        {rowLabels && (
          <div className="flex flex-none flex-col pr-1 font-mono text-[9px] leading-none text-muted" style={{ height: rowHeight ? rows * rowHeight : undefined }}>
            {rowLabels.map((label, i) => (
              <span key={i} className={cn("flex flex-1 items-center justify-end", hover?.r === i && "text-ink")}>
                {rows <= 48 ? label : ""}
              </span>
            ))}
          </div>
        )}
        <canvas
          ref={ref}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          className="min-w-0 flex-1 rounded-sm bg-surface-3/40 [image-rendering:pixelated]"
          style={{ height: rowHeight ? rows * rowHeight : undefined, aspectRatio: rowHeight ? undefined : `${cols} / ${rows}` }}
        />
      </div>
      {colLabels && cols <= 48 && (
        <div className="flex font-mono text-[9px] leading-none text-muted" style={{ paddingLeft: rowLabels ? 14 : 0 }}>
          {colLabels.map((label, i) => (
            <span key={i} className={cn("flex-1 pt-1 text-center", hover?.c === i && "text-ink")}>
              {label}
            </span>
          ))}
        </div>
      )}
      {hover && (
        <div className="pointer-events-none absolute z-10 rounded-md border border-line bg-surface px-2 py-1 font-mono text-[11px] whitespace-nowrap shadow-lg" style={{ left: Math.min(hover.x + 14, 220), top: hover.y + 14 }}>
          {describe ? describe(hover.r, hover.c, at(hover.r, hover.c)) : `${hover.r}, ${hover.c}: ${at(hover.r, hover.c).toFixed(3)}`}
        </div>
      )}
    </div>
  );
}

/** Feature maps [H, W, C] as a grid of small pictures, one per channel. */
function Maps({ layer, image }: { layer: Layer; image: boolean }) {
  const [h, w, c] = layer.shape;
  const data = layer.data;
  if (image) {
    return <Picture data={layer.picture ?? data} h={h} w={w} c={c} />;
  }
  const show = Math.min(c, 24);
  return (
    <div>
      <div className="grid grid-cols-6 gap-1 sm:grid-cols-8">
        {Array.from({ length: show }, (_, k) => (
          <Heatmap key={k} rows={h} cols={w} at={(r, col) => data[(r * w + col) * c + k]} scale={absMax(data, k, data.length, c)} describe={(r, col, v) => `channel ${k + 1}, pixel (${r}, ${col}): ${v.toFixed(3)}`} />
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-muted">
        {show < c ? `First ${show} of ${c} channels. ` : `All ${c} channels. `}Each is one filter&apos;s answer to &ldquo;is my pattern here?&rdquo;: pink where yes, blue where the opposite.
      </p>
    </div>
  );
}

function Picture({ data, h, w, c }: { data: Float32Array; h: number; w: number; c: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const image = ctx.createImageData(w, h);
    for (let p = 0; p < h * w; p++) {
      for (let k = 0; k < 3; k++) image.data[p * 4 + k] = Math.round(Math.max(0, Math.min(1, data[p * c + Math.min(k, c - 1)])) * 255);
      image.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [data, h, w, c]);
  return (
    <div className="flex items-end gap-3">
      <canvas ref={ref} className="size-28 rounded-md border border-line [image-rendering:pixelated]" />
      <p className="text-[11px] text-muted">
        What the model actually sees: {h}×{w} pixels{c === 1 ? ", grey" : ", red, green and blue"}, each a number from 0 to 1.
      </p>
    </div>
  );
}

function Bars({ data, labels, colour = "var(--accent)", max = 96 }: { data: Float32Array | number[]; labels?: string[]; colour?: string; max?: number }) {
  const shown = Math.min(data.length, max);
  let scale = 0;
  for (let i = 0; i < shown; i++) scale = Math.max(scale, Math.abs(data[i]));
  return (
    <div>
      <div className="flex h-16 items-center gap-px">
        {Array.from({ length: shown }, (_, i) => {
          const v = data[i] / (scale || 1);
          return (
            <div key={i} className="relative h-full flex-1" title={`${labels?.[i] ?? i}: ${data[i].toFixed(3)}`}>
              <div className="absolute inset-x-0 rounded-[1px]" style={{ background: v >= 0 ? colour : "#5fb4ff", top: v >= 0 ? `${50 - v * 50}%` : "50%", height: `${Math.abs(v) * 50}%` }} />
            </div>
          );
        })}
      </div>
      {shown < data.length && <p className="mt-1 text-[11px] text-muted">First {shown} of {data.length} numbers.</p>}
    </div>
  );
}

/** A histogram of the values, with the input's shown faintly behind for activations. */
function Histogram({ data, before }: { data: Float32Array; before?: Float32Array }) {
  const bins = 32;
  const { lo, hi } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const set of [data, before]) if (set) for (let i = 0; i < set.length; i++) { if (set[i] < lo) lo = set[i]; if (set[i] > hi) hi = set[i]; }
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    return { lo, hi };
  }, [data, before]);
  const count = (set: Float32Array) => {
    const out = new Array(bins).fill(0);
    for (let i = 0; i < set.length; i++) out[Math.min(bins - 1, Math.floor(((set[i] - lo) / (hi - lo)) * bins))]++;
    return out;
  };
  const after = count(data);
  const prior = before ? count(before) : null;
  const top = Math.max(...after, ...(prior ?? [0]));
  const zero = ((0 - lo) / (hi - lo)) * 100;
  return (
    <div>
      <div className="relative flex h-10 items-end gap-px">
        {after.map((n, i) => (
          <div key={i} className="relative h-full flex-1">
            {prior && <div className="absolute inset-x-0 bottom-0 bg-muted/25" style={{ height: `${(prior[i] / top) * 100}%` }} />}
            <div className="absolute inset-x-0 bottom-0 bg-accent/80" style={{ height: `${(n / top) * 100}%` }} />
          </div>
        ))}
        {zero > 0 && zero < 100 && <div className="absolute inset-y-0 w-px bg-ink/40" style={{ left: `${zero}%` }} />}
      </div>
      <div className="mt-0.5 flex justify-between font-mono text-[10px] text-muted">
        <span>{lo.toFixed(2)}</span>
        {prior && <span>grey: before · pink: after</span>}
        <span>{hi.toFixed(2)}</span>
      </div>
    </div>
  );
}

function stats(data: Float32Array) {
  let sum = 0;
  let zeros = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (data[i] === 0) zeros++;
  }
  const mean = sum / data.length;
  let sq = 0;
  for (let i = 0; i < data.length; i++) sq += (data[i] - mean) ** 2;
  return { mean, std: Math.sqrt(sq / data.length), zeros: zeros / data.length };
}

function softmax(values: ArrayLike<number>) {
  let m = -Infinity;
  for (let i = 0; i < values.length; i++) m = Math.max(m, values[i]);
  const e = Array.from(values, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function topK(probs: number[], k: number) {
  return probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
}

// -- What each layer shows -----------------------------------------------------------

function Attention({ layer }: { layer: Layer }) {
  const [heads, q, k] = layer.attention!.shape;
  const data = layer.attention!.data;
  const [head, setHead] = useState(0);
  const rows = layer.rows ?? [];
  const keys = layer.keys ?? [];
  const at = (h: number) => (r: number, c: number) => data[(h * q + r) * k + c];
  const limit = Math.min(q, 48);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {Array.from({ length: heads }, (_, h) => (
          <button key={h} type="button" onClick={() => setHead(h)} className={cn("w-14 rounded-md border p-1 text-left", h === head ? "border-accent" : "border-line hover:border-muted")}>
            <Heatmap rows={Math.min(q, 24)} cols={Math.min(k, 24)} at={at(h)} scale={1} sequential />
            <span className="mt-0.5 block text-center font-mono text-[9.5px] text-muted">head {h + 1}</span>
          </button>
        ))}
      </div>
      <Heatmap
        rows={limit}
        cols={Math.min(k, 48)}
        at={at(head)}
        scale={1}
        sequential
        rowLabels={rows.slice(0, limit)}
        colLabels={keys.slice(0, Math.min(k, 48))}
        describe={(r, c, v) => `"${rows[r]}" looks at "${keys[c]}": ${(v * 100).toFixed(1)}%`}
      />
      <p className="text-[11px] text-muted">
        Head {head + 1}: each row is a token, each column a token it can look at, and brightness is how much of that token&apos;s value it takes. Rows add up to 100%.
        {layer.node.op === "attention" && (layer.node.args.causal === undefined || layer.node.args.causal === true || layer.node.args.causal === "true") ? " The empty upper triangle is the causal mask: no peeking ahead." : ""}
      </p>
    </div>
  );
}

function Sequence({ layer, used }: { layer: Layer; used: number }) {
  const [s, d] = layer.shape;
  const rows = Math.min(s, Math.max(used, 1), 48);
  const cols = Math.min(d, 96);
  const data = layer.data;
  const scale = useMemo(() => absMax(data, 0, rows * d), [data, rows, d]);
  return (
    <div>
      <Heatmap rows={rows} cols={cols} at={(r, c) => data[r * d + c]} scale={scale} rowLabels={layer.rows?.slice(0, rows)} rowHeight={rows > 24 ? 6 : 10} describe={(r, c, v) => `"${layer.rows?.[r] ?? r}", feature ${c}: ${v.toFixed(3)}`} />
      <p className="mt-1.5 text-[11px] text-muted">
        One row per token{rows < s ? ` (first ${rows} of ${s})` : ""}, {cols < d ? `first ${cols} of ${d}` : d} numbers each. Pink is positive, blue negative.
      </p>
    </div>
  );
}

export function Output({ layer, trace }: { layer: Layer; trace: Trace }) {
  const { task } = trace;
  const data = layer.data;
  if (layer.shape.length === 1 && trace.classes) {
    const probs = softmax(data);
    const best = topK(probs, Math.min(5, probs.length));
    return <Ranked items={best.map(({ p, i }) => ({ label: trace.classes![i] ?? String(i), p }))} />;
  }
  if (layer.shape.length !== 2) return null;
  const [s, v] = layer.shape;
  const row = (r: number) => softmax(data.subarray(r * v, (r + 1) * v));
  const decode = (i: number) => {
    if (trace.decode) return trace.decode(i);
    if (task.kind === "seq2seq") return i === 0 ? "▸" : "0123456789abcdef"[i - 1] ?? "?";
    if (task.kind === "language-modelling" || task.kind === "masked-tokens") return i === 0 ? "↵" : i >= 1 && i <= 95 ? (i === 1 ? "␣" : String.fromCharCode(i + 31)) : "?";
    return String(i);
  };
  if (task.kind === "language-modelling") {
    const last = Math.min(trace.used, s) - 1;
    const probs = row(last);
    return (
      <div className="space-y-2">
        <p className="text-[12px]">
          After <span className="font-mono">&ldquo;…{(layer.rows ?? []).slice(Math.max(0, last - 11), last + 1).join("")}&rdquo;</span> the model expects:
        </p>
        <Ranked items={topK(probs, 5).map(({ p, i }) => ({ label: decode(i), p }))} mono />
      </div>
    );
  }
  const guesses = Array.from({ length: Math.min(s, trace.used, 48) }, (_, r) => topK(row(r), 1)[0]);
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-0.5 font-mono text-[13px]">
        {guesses.map((g, r) => (
          <span key={r} className="grid w-5 place-items-center rounded-sm py-0.5" style={{ background: `color-mix(in oklab, var(--accent) ${Math.round(g.p * 70)}%, transparent)` }} title={`${(g.p * 100).toFixed(0)}% sure`}>
            {decode(g.i)}
          </span>
        ))}
      </div>
      <p className="text-[11px] text-muted">The most likely token at each position; stronger colour means more confident.</p>
    </div>
  );
}

function Ranked({ items, mono }: { items: { label: string; p: number }[]; mono?: boolean }) {
  return (
    <ol className="space-y-1">
      {items.map((item, i) => (
        <li key={i} className="flex items-center gap-2 text-[12px]">
          <span className={cn("w-20 truncate", mono && "font-mono text-[13px]", i === 0 && "font-semibold")}>{item.label}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
            <span className="block h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${Math.max(1, item.p * 100)}%` }} />
          </span>
          <span className="w-12 text-right font-mono text-[11px] text-muted">{(item.p * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ol>
  );
}

const QUIET = new Set(["dropout", "flatten", "relu", "relu6", "gelu", "silu", "sigmoid", "tanh", "hardswish", "take_first", "mean_pool", "gap"]);

/** True once the element has come near the screen, so long traces only draw what is looked at. */
function useSeen<T extends Element>() {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element || seen) return;
    const observer = new IntersectionObserver((entries) => entries.some((entry) => entry.isIntersecting) && setSeen(true), { rootMargin: "800px 0px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, [seen]);
  return [ref, seen] as const;
}

function LayerCard({ layer, index, trace, last, state, onFocus }: { layer: Layer; index: number; trace: Trace; last: boolean; state: "past" | "now" | "later"; onFocus: () => void }) {
  const [open, setOpen] = useState(!QUIET.has(layer.node.op));
  const [ref, seen] = useSeen<HTMLElement>();
  const words = useMemo(() => explain(layer.node, layer.inputShapes, layer.outputShape), [layer.node, layer.inputShapes, layer.outputShape]);
  const numbers = useMemo(() => stats(layer.data), [layer.data]);
  const category = OPS[layer.node.op]?.category ?? "reshape";
  const colour = CATEGORY_COLOURS[category];
  const isInput = layer.node.op === "input";
  const visual = () => {
    if (last) return <Output layer={layer} trace={trace} />;
    if (layer.attention) return <Attention layer={layer} />;
    if (layer.shape.length === 3) return <Maps layer={layer} image={isInput} />;
    if (layer.shape.length === 2) return <Sequence layer={layer} used={trace.used} />;
    if (layer.shape.length === 1 && isInput) return null;
    if (layer.shape.length === 1) return <Bars data={layer.data} colour={colour} />;
    return null;
  };
  return (
    <article
      ref={ref}
      data-step={index}
      onClick={onFocus}
      className={cn(
        "relative rounded-xl border bg-surface transition-[opacity,border-color,box-shadow] duration-500",
        state === "now" ? "border-accent shadow-[0_0_0_4px_color-mix(in_oklab,var(--accent)_18%,transparent)]" : "border-line",
        state === "later" && "opacity-35",
      )}
    >
      <header className="flex items-center gap-2 px-3 pt-2.5">
        <span className="size-2.5 flex-none rounded-full" style={{ background: colour }} />
        <span className="text-[13px] font-semibold">{layer.node.label ?? layer.node.op}</span>
        <span className="font-mono text-[11px] text-muted">{layer.node.op}</span>
        <span className="ml-auto font-mono text-[11px] text-muted">[{layer.outputShape.join("×")}]</span>
        <button type="button" onClick={(event) => { event.stopPropagation(); setOpen(!open); }} className="rounded px-1 text-[11px] text-muted hover:text-ink">
          {open ? "less" : "more"}
        </button>
      </header>
      <div className="px-3 pb-3">
        <div className="mt-1 text-[14px]">
          <Tex tex={words.tex} display={open} />
        </div>
        {open && (
          <>
            {words.numbers && (
              <div className="rounded-md bg-surface-2 px-2 text-[13px] text-muted">
                <Tex tex={words.numbers} display />
              </div>
            )}
            <p className="mt-2 text-[12.5px] leading-relaxed">{words.words}</p>
            <div className="mt-3">{seen ? visual() : <div className="h-24 rounded-md bg-surface-2" />}</div>
            {seen && !isInput && !last && (
              <div className="mt-3 grid grid-cols-[1fr_auto] items-end gap-3">
                <Histogram data={layer.data} before={layer.before} />
                <dl className="grid grid-cols-2 gap-x-2 font-mono text-[10.5px] text-muted">
                  <dt>mean</dt>
                  <dd className="text-right text-ink">{numbers.mean.toFixed(3)}</dd>
                  <dt>spread</dt>
                  <dd className="text-right text-ink">{numbers.std.toFixed(3)}</dd>
                  <dt>zeros</dt>
                  <dd className="text-right text-ink">{(numbers.zeros * 100).toFixed(0)}%</dd>
                </dl>
              </div>
            )}
          </>
        )}
      </div>
    </article>
  );
}

// -- One model's column --------------------------------------------------------------

function Column({ slot, inputs, cursor, playing, onSteps, onSeek }: { slot: SlotIndex; inputs: Inputs; cursor: number; playing: boolean; onSteps: (n: number) => void; onSeek: (fraction: number) => void }) {
  const data = useLab((state) => state.slots[slot]);
  const { value, error } = usePreparedModel(slot, data?.model);
  const [trace, setTrace] = useState<{ value: Trace | null; error: string | null }>({ value: null, error: null });
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!value) return;
    let live = true;
    const timer = window.setTimeout(() => {
      void import("tessera-lab").then((lab) => {
        if (!live) return;
        try {
          setTrace({ value: runTrace(lab, value, inputs), error: null });
        } catch (caught) {
          setTrace({ value: null, error: caught instanceof Error ? caught.message : String(caught) });
        }
      });
    }, 120);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [value, inputs]);

  const layers = trace.value?.layers ?? [];
  const now = layers.length ? Math.round(cursor * (layers.length - 1)) : 0;

  useEffect(() => onSteps(layers.length), [layers.length, onSteps]);
  useEffect(() => {
    if (!playing) return;
    list.current?.querySelector(`[data-step="${now}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [now, playing]);

  if (!data) return null;
  const output = layers[layers.length - 1];
  return (
    <section className={cn("flex min-h-0 flex-col", slot === 1 && "border-t border-line md:border-t-0 md:border-l")}>
      <div className="flex flex-none flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-surface px-4 py-2.5">
        <span className="grid size-5 place-items-center rounded text-[11px] font-semibold text-[#0b0d12]" style={{ background: SLOT_COLOURS[slot] }}>
          {slot ? "B" : "A"}
        </span>
        <span className="text-[13px] font-medium">{data.model.name}</span>
        {value && (
          <>
            <WeightsBadge weights={value.weights} />
            <RealWeightsButton slot={slot} model={data.model} weights={value.weights} />
          </>
        )}
        {trace.value && <span className="ml-auto font-mono text-[11px] text-muted">{layers.length} steps · {trace.value.ms.toFixed(0)} ms</span>}
        {value && <p className="w-full text-[11.5px] text-muted">{value.prepared.description}</p>}
      </div>
      {(error || trace.error) && <p className="m-4 rounded-lg border border-danger/40 bg-danger/10 p-3 text-[12.5px] text-danger">{error ?? trace.error}</p>}
      {!error && !trace.value && !trace.error && <p className="p-6 text-center text-[13px] text-muted">Running…</p>}
      {trace.value && (
        <div className="flex min-h-0 flex-1">
          {/* The whole pass at a glance: one tick per step, in its category colour. Click to jump. */}
          <div className="flex w-3 flex-none flex-col gap-px py-3 pl-1.5">
            {layers.map((layer, i) => (
              <button
                key={layer.key}
                type="button"
                aria-label={`Step ${i + 1}: ${layer.node.label ?? layer.node.op}`}
                onClick={() => {
                  onSeek(layers.length > 1 ? i / (layers.length - 1) : 0);
                  list.current?.querySelector(`[data-step="${i}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
                }}
                className={cn("min-h-px flex-1 rounded-full transition-opacity", i > now && "opacity-30")}
                style={{ background: i === now ? "var(--ink)" : CATEGORY_COLOURS[OPS[layer.node.op]?.category ?? "reshape"] }}
              />
            ))}
          </div>
          <div ref={list} className="scroll-quiet min-h-0 flex-1 space-y-0 overflow-y-auto px-3 py-3">
            {output && trace.value.task && (
              <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3">
                <p className="mb-2 text-[11px] font-medium tracking-wide text-muted uppercase">{trace.value.task.label}</p>
                <Output layer={output} trace={trace.value} />
              </div>
            )}
            {layers.map((layer, i) => {
              const header = layer.block && layer.block !== layers[i - 1]?.block ? layer.block : null;
              return (
                <div key={layer.key}>
                  {header && <p className="mt-4 mb-2 text-[11px] font-medium tracking-wide text-muted uppercase">{header}</p>}
                  {i > 0 && !header && (
                    <div className="ml-5 flex h-5 items-center gap-2 border-l border-dashed border-line pl-3">
                      <span className="font-mono text-[10px] text-muted">{layers[i - 1].outputShape.join("×")}</span>
                    </div>
                  )}
                  <LayerCard layer={layer} index={i} trace={trace.value!} last={i === layers.length - 1} state={i < now ? "past" : i === now ? "now" : "later"} onFocus={() => onSeek(layers.length > 1 ? i / (layers.length - 1) : 0)} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

// -- Input pads ----------------------------------------------------------------------

export function DrawPad({ onChange, setPad }: { onChange: () => void; setPad: (canvas: HTMLCanvasElement) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef<{ x: number; y: number } | null>(null);
  const last = useRef(0);
  const [glyph, setGlyph] = useState(3);

  useEffect(() => {
    const canvas = ref.current!;
    canvas.width = PAD;
    canvas.height = PAD;
    drawGlyph(canvas, "3");
    setPad(canvas);
    onChange();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const point = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return { x: ((event.clientX - box.left) / box.width) * PAD, y: ((event.clientY - box.top) / box.height) * PAD };
  };
  const stroke = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const ctx = ref.current!.getContext("2d")!;
    const p = point(event);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = PAD / 11;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(drawing.current.x, drawing.current.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    drawing.current = p;
    if (performance.now() - last.current > 140) {
      last.current = performance.now();
      onChange();
    }
  };
  const upload = (file: File) => {
    const img = new Image();
    img.onload = () => {
      const ctx = ref.current!.getContext("2d")!;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, PAD, PAD);
      const s = Math.min(PAD / img.width, PAD / img.height);
      ctx.drawImage(img, (PAD - img.width * s) / 2, (PAD - img.height * s) / 2, img.width * s, img.height * s);
      URL.revokeObjectURL(img.src);
      onChange();
    };
    img.src = URL.createObjectURL(file);
  };
  return (
    <div className="flex items-center gap-2">
      <canvas
        ref={ref}
        aria-label="Drawing pad"
        className="size-[84px] cursor-crosshair touch-none rounded-lg border border-line"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          drawing.current = point(event);
        }}
        onPointerMove={stroke}
        onPointerUp={() => {
          drawing.current = null;
          onChange();
        }}
      />
      <div className="flex flex-col gap-1 text-[11.5px]">
        <button type="button" className="rounded-md border border-line px-2 py-0.5 hover:border-muted" onClick={() => { const ctx = ref.current!.getContext("2d")!; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, PAD, PAD); onChange(); }}>
          Clear
        </button>
        <button type="button" className="rounded-md border border-line px-2 py-0.5 hover:border-muted" onClick={() => { const next = (glyph + 1) % 10; setGlyph(next); drawGlyph(ref.current!, String(next)); onChange(); }}>
          Next digit
        </button>
        <label className="flex cursor-pointer items-center gap-1 rounded-md border border-line px-2 py-0.5 hover:border-muted">
          <Upload className="size-3" />
          Photo
          <input type="file" accept="image/*" className="hidden" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])} />
        </label>
      </div>
    </div>
  );
}

// -- The view ------------------------------------------------------------------------

export default function TraceView() {
  const slots = useLab((state) => state.slots);
  const [text, setText] = useState(DEFAULT_TEXT);
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [pad, setPad] = useState<HTMLCanvasElement | null>(null);
  const [padVersion, setPadVersion] = useState(0);
  const [cursor, setCursor] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [steps, setSteps] = useState<[number, number]>([0, 0]);
  const shown = ([0, 1] as SlotIndex[]).filter((index) => slots[index]);
  const inputs = useMemo<Inputs>(() => ({ text, symbols, pad, padVersion }), [text, symbols, pad, padVersion]);
  const onSteps = useMemo(() => [0, 1].map((slot) => (n: number) => setSteps((current) => (current[slot] === n ? current : ((slot ? [current[0], n] : [n, current[1]]) as [number, number])))), []);

  // What the loaded models read decides which inputs show.
  const kinds = useMemo(() => {
    const out = new Set<string>();
    for (const slot of slots) {
      if (!slot) continue;
      const input = slot.model.nodes.find((node) => node.op === "input");
      const shape = input ? slot.analysis.info.get(input.id)?.shape : undefined;
      if (shape?.length === 3) out.add("image");
      else if (slot.model.nodes.some((node) => node.op === "cross_attention") || slot.model.nodes.filter((node) => node.op === "input").length > 1) out.add("symbols");
      else if (slot.model.nodes.some((node) => node.op === "lmhead")) out.add("text");
      else out.add("text-or-symbols");
    }
    return out;
  }, [slots]);

  const longest = Math.max(...steps, 1);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setCursor((c) => {
        const next = c + 1 / Math.max(1, longest - 1);
        if (next >= 1) {
          setPlaying(false);
          return 1;
        }
        return next;
      });
    }, 650);
    return () => window.clearInterval(timer);
  }, [playing, longest]);

  const step = (direction: number) => {
    setPlaying(false);
    setCursor((c) => Math.max(0, Math.min(1, c + direction / Math.max(1, longest - 1))));
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-none flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-surface-2 px-4 py-2.5">
        {kinds.has("image") && <DrawPad setPad={setPad} onChange={() => setPadVersion((v) => v + 1)} />}
        {(kinds.has("text") || kinds.has("text-or-symbols")) && (
          <label className="min-w-[220px] flex-1 text-[11.5px] text-muted">
            Text in{kinds.has("text-or-symbols") ? " (use _ to hide a character for an encoder to fill in)" : ""}
            <input value={text} onChange={(event) => setText(event.target.value)} className="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1.5 font-mono text-[13px] text-ink" spellCheck={false} />
          </label>
        )}
        {(kinds.has("symbols") || kinds.has("text-or-symbols")) && (
          <label className="w-56 text-[11.5px] text-muted">
            Symbols in (0-9, a-f)
            <input value={symbols} onChange={(event) => setSymbols(event.target.value)} className="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1.5 font-mono text-[13px] tracking-widest text-ink" spellCheck={false} />
          </label>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button type="button" aria-label="Back to the start" onClick={() => { setPlaying(false); setCursor(0); }} className="grid size-8 place-items-center rounded-lg border border-line hover:border-muted">
            <RotateCcw className="size-3.5" />
          </button>
          <button type="button" aria-label="Previous layer" onClick={() => step(-1)} className="grid size-8 place-items-center rounded-lg border border-line hover:border-muted">
            <SkipBack className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              if (!playing && cursor >= 1) setCursor(0);
              setPlaying(!playing);
            }}
            className="flex h-8 items-center gap-1.5 rounded-lg bg-ink px-3 text-[12.5px] font-medium text-bg"
          >
            {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            {playing ? "Pause" : "Play the pass"}
          </button>
          <button type="button" aria-label="Next layer" onClick={() => step(1)} className="grid size-8 place-items-center rounded-lg border border-line hover:border-muted">
            <SkipForward className="size-3.5" />
          </button>
        </div>
      </div>
      <div className={cn("grid min-h-0 flex-1", shown.length === 2 && "grid-rows-2 md:grid-cols-2 md:grid-rows-1")}>
        {shown.map((slot) => (
          <Column key={slot} slot={slot} inputs={inputs} cursor={cursor} playing={playing} onSteps={onSteps[slot]} onSeek={(fraction) => { setPlaying(false); setCursor(fraction); }} />
        ))}
      </div>
    </div>
  );
}
