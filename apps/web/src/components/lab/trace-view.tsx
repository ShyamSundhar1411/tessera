"use client";

import { Box, Pause, Play, RotateCcw, SkipBack, SkipForward, Upload } from "lucide-react";
import dynamic from "next/dynamic";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { OPS, type Model, type Node, type Shape } from "tessera-engine";
import type { Gpt2Tokenizer, GraphModel, PlainFeed, PreparedModel, TaskSpec, TracedLayer } from "tessera-lab";
import { CATEGORY_COLOURS } from "@/lib/display";
import { explain } from "@/lib/equations";
import { insight, working } from "@/lib/trace-insight";
import { useLab, type SlotIndex } from "@/lib/store";
import { useTraceShared } from "@/lib/trace-shared";
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

const Graph3D = dynamic(() => import("./graph-3d"), { ssr: false, loading: () => <div className="grid h-full place-items-center text-[13px] text-muted">Loading 3D…</div> });

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
  /** The graph that produced it, for reading weights and the logit lens against the same model. */
  graph: GraphModel;
}

/** What the reader has picked in a column, shared by every card in it. */
interface Interact {
  /** The token every sequence card is about. */
  token: number;
  setToken(token: number): void;
  /** One number whose working is shown. */
  picked: { key: string; index: number } | null;
  pick(key: string, index: number): void;
  byKey: Map<string, Layer>;
  weights(key: string): Record<string, { shape: number[]; data: Float32Array }>;
  /** What the model would predict from each residual-stream layer, at the picked token. */
  lens: Map<string, { id: number; p: number }[]>;
  decode(id: number): string;
}

const InteractContext = createContext<Interact | null>(null);

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
  return { layers, ms, used, task: prepared.task, classes, decode: tokenizer ? (id: number) => showToken(tokenizer.decode([id])) : undefined, graph };
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
  /** Clicking a cell picks it. */
  onPick?: (r: number, c: number) => void;
  /** A row to outline, such as the picked token. */
  highlightRow?: number;
  /** A cell to ring, such as the number whose working is open. */
  pickedCell?: { r: number; c: number };
}

function Heatmap({ rows, cols, at, scale, sequential, rowLabels, colLabels, describe, className, rowHeight, onPick, highlightRow, pickedCell }: HeatmapProps) {
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
              <span key={i} className={cn("flex flex-1 items-center justify-end", (hover?.r === i || highlightRow === i) && "font-semibold text-ink")}>
                {rows <= 48 ? label : ""}
              </span>
            ))}
          </div>
        )}
        <div className="relative min-w-0 flex-1">
          <canvas
            ref={ref}
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
            onClick={(event) => {
              if (!onPick) return;
              const box = event.currentTarget.getBoundingClientRect();
              const c = Math.floor(((event.clientX - box.left) / box.width) * cols);
              const r = Math.floor(((event.clientY - box.top) / box.height) * rows);
              if (r < 0 || r >= rows || c < 0 || c >= cols) return;
              event.stopPropagation();
              onPick(r, c);
            }}
            className={cn("block w-full rounded-sm bg-surface-3/40 [image-rendering:pixelated]", onPick && "cursor-pointer")}
            style={{ height: rowHeight ? rows * rowHeight : undefined, aspectRatio: rowHeight ? undefined : `${cols} / ${rows}` }}
          />
          {highlightRow !== undefined && highlightRow < rows && (
            <div className="pointer-events-none absolute inset-x-0 rounded-sm ring-1 ring-ink/80" style={{ top: `${(highlightRow / rows) * 100}%`, height: `${100 / rows}%` }} />
          )}
          {pickedCell && pickedCell.r < rows && pickedCell.c < cols && (
            <div className="pointer-events-none absolute rounded-[2px] ring-2 ring-[#f7b955]" style={{ top: `${(pickedCell.r / rows) * 100}%`, left: `${(pickedCell.c / cols) * 100}%`, height: `${100 / rows}%`, width: `${100 / cols}%`, minWidth: 4, minHeight: 4 }} />
          )}
        </div>
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

/** Feature maps [H, W, C] as a grid of small pictures, one per channel, strongest first. Click a pixel to see how it was computed. */
function Maps({ layer, image }: { layer: Layer; image: boolean }) {
  const ctx = useContext(InteractContext);
  const [h, w, c] = layer.shape;
  const data = layer.data;
  const order = useMemo(() => {
    const strength = Array.from({ length: c }, (_, k) => {
      let s = 0;
      for (let p = 0; p < h * w; p++) s += Math.abs(data[p * c + k]);
      return s;
    });
    return strength.map((_, k) => k).sort((a, b) => strength[b] - strength[a]);
  }, [data, h, w, c]);
  if (image) {
    return <Picture data={layer.picture ?? data} h={h} w={w} c={c} />;
  }
  const show = Math.min(c, 24);
  const picked = ctx?.picked?.key === layer.key ? ctx.picked.index : -1;
  return (
    <div>
      <div className="grid grid-cols-6 gap-1 sm:grid-cols-8">
        {order.slice(0, show).map((k) => (
          <div key={k}>
            <Heatmap
              rows={h}
              cols={w}
              at={(r, col) => data[(r * w + col) * c + k]}
              scale={absMax(data, k, data.length, c)}
              describe={(r, col, v) => `filter ${k + 1}, pixel (${r + 1}, ${col + 1}): ${v.toFixed(3)}. Click to see how.`}
              onPick={ctx ? (r, col) => ctx.pick(layer.key, (r * w + col) * c + k) : undefined}
              pickedCell={picked >= 0 && picked % c === k ? { r: Math.floor(picked / c / w), c: Math.floor(picked / c) % w } : undefined}
            />
            <span className="block text-center font-mono text-[9px] text-muted">#{k + 1}</span>
          </div>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-muted">
        {show < c ? `The ${show} strongest of ${c} filters. ` : `All ${c} filters, strongest first. `}Each is one filter&apos;s answer to &ldquo;is my pattern here?&rdquo;: pink where yes, blue where the opposite. Click any pixel to see the sum that made it.
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

function Bars({ data, labels, colour = "var(--accent)", max = 96, onPick }: { data: Float32Array | number[]; labels?: string[]; colour?: string; max?: number; onPick?: (index: number) => void }) {
  const shown = Math.min(data.length, max);
  let scale = 0;
  for (let i = 0; i < shown; i++) scale = Math.max(scale, Math.abs(data[i]));
  return (
    <div>
      <div className="flex h-16 items-center gap-px">
        {Array.from({ length: shown }, (_, i) => {
          const v = data[i] / (scale || 1);
          return (
            <div
              key={i}
              className={cn("relative h-full flex-1", onPick && "cursor-pointer hover:bg-ink/10")}
              title={`${labels?.[i] ?? i}: ${data[i].toFixed(3)}${onPick ? ". Click to see how." : ""}`}
              onClick={(event) => {
                if (!onPick) return;
                event.stopPropagation();
                onPick(i);
              }}
            >
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
  const ctx = useContext(InteractContext);
  const [heads, q, k] = layer.attention!.shape;
  const data = layer.attention!.data;
  const [head, setHead] = useState(0);
  const rows = layer.rows ?? [];
  const keys = layer.keys ?? [];
  const at = (h: number) => (r: number, c: number) => data[(h * q + r) * k + c];
  const limit = Math.min(q, 48);
  const token = Math.min(ctx?.token ?? q - 1, q - 1);
  // The picked token's attention in this head, biggest first.
  const focus = Array.from({ length: k }, (_, j) => ({ label: keys[j] ?? String(j), p: data[(head * q + token) * k + j] }))
    .filter((item) => item.p > 0)
    .sort((a, b) => b.p - a.p)
    .slice(0, 6);
  return (
    <div className="space-y-3">
      <div className="rounded-lg bg-surface-2 p-2.5">
        <p className="mb-1.5 text-[12px]">
          In head {head + 1}, <span className="rounded bg-ink/10 px-1 font-mono">{rows[token]}</span> takes its information from:
        </p>
        <Ranked items={focus} mono />
      </div>
      <div className="flex flex-wrap gap-1.5">
        {Array.from({ length: heads }, (_, h) => (
          <button key={h} type="button" onClick={(event) => { event.stopPropagation(); setHead(h); }} className={cn("w-14 rounded-md border p-1 text-left", h === head ? "border-accent" : "border-line hover:border-muted")}>
            <Heatmap rows={Math.min(q, 24)} cols={Math.min(k, 24)} at={at(h)} scale={1} sequential highlightRow={token < 24 ? token : undefined} />
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
        highlightRow={token}
        onPick={ctx ? (r) => ctx.setToken(r) : undefined}
        describe={(r, c, v) => `"${rows[r]}" looks at "${keys[c]}": ${(v * 100).toFixed(1)}%. Click to follow "${rows[r]}".`}
      />
      <p className="text-[11px] text-muted">
        Each row is a token, each column a token it can look at, and brightness is how much it takes from it. Rows add up to 100%. Click a row to follow that token through every layer.
        {layer.node.op === "attention" && (layer.node.args.causal === undefined || layer.node.args.causal === true || layer.node.args.causal === "true") ? " The empty upper triangle is the causal mask: no peeking ahead." : ""}
      </p>
    </div>
  );
}

function Sequence({ layer, used }: { layer: Layer; used: number }) {
  const ctx = useContext(InteractContext);
  const [s, d] = layer.shape;
  const rows = Math.min(s, Math.max(used, 1), 48);
  const cols = Math.min(d, 96);
  const data = layer.data;
  const scale = useMemo(() => absMax(data, 0, rows * d), [data, rows, d]);
  const picked = ctx?.picked?.key === layer.key ? ctx.picked.index : -1;
  return (
    <div>
      <Heatmap
        rows={rows}
        cols={cols}
        at={(r, c) => data[r * d + c]}
        scale={scale}
        rowLabels={layer.rows?.slice(0, rows)}
        rowHeight={rows > 24 ? 6 : 10}
        highlightRow={ctx?.token}
        pickedCell={picked >= 0 ? { r: Math.floor(picked / d), c: picked % d } : undefined}
        onPick={
          ctx
            ? (r, c) => {
                ctx.setToken(r);
                ctx.pick(layer.key, r * d + c);
              }
            : undefined
        }
        describe={(r, c, v) => `"${layer.rows?.[r] ?? r}", number ${c + 1}: ${v.toFixed(3)}. Click to see how it was made.`}
      />
      <p className="mt-1.5 text-[11px] text-muted">
        One row per token{rows < s ? ` (first ${rows} of ${s})` : ""}, {cols < d ? `first ${cols} of ${d}` : d} numbers each: the token&apos;s meaning so far, as the model stores it. Click a number to see how it was computed.
      </p>
    </div>
  );
}

export function Output({ layer, trace }: { layer: Layer; trace: Trace }) {
  const ctx = useContext(InteractContext);
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
    const last = Math.min(ctx?.token ?? trace.used - 1, s - 1);
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
          <button
            key={r}
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              ctx?.setToken(r);
            }}
            className={cn("grid w-5 place-items-center rounded-sm py-0.5", ctx?.token === r && "ring-1 ring-ink")}
            style={{ background: `color-mix(in oklab, var(--accent) ${Math.round(g.p * 70)}%, transparent)` }}
            title={`${(g.p * 100).toFixed(0)}% sure`}
          >
            {decode(g.i)}
          </button>
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
  const ctx = useContext(InteractContext);
  const inputs = useMemo(() => layer.inputs.map((key) => ctx?.byKey.get(key)).filter((item): item is Layer => !!item), [layer.inputs, ctx?.byKey]);
  const said = useMemo(() => insight(layer, inputs, ctx?.token ?? 0), [layer, inputs, ctx?.token]);
  const lens = ctx?.lens.get(layer.key);
  const picked = ctx?.picked?.key === layer.key ? ctx.picked.index : null;
  const work = useMemo(() => (picked === null || !ctx ? null : working(layer, inputs, picked, () => ctx.weights(layer.key))), [picked, layer, inputs, ctx]);
  const category = OPS[layer.node.op]?.category ?? "reshape";
  const colour = CATEGORY_COLOURS[category];
  const isInput = layer.node.op === "input";
  const visual = () => {
    if (last) return <Output layer={layer} trace={trace} />;
    if (layer.attention) return <Attention layer={layer} />;
    if (layer.shape.length === 3) return <Maps layer={layer} image={isInput} />;
    if (layer.shape.length === 2) return <Sequence layer={layer} used={trace.used} />;
    if (layer.shape.length === 1 && isInput) return null;
    if (layer.shape.length === 1) return <Bars data={layer.data} colour={colour} onPick={ctx ? (i) => ctx.pick(layer.key, i) : undefined} />;
    return null;
  };
  return (
    <article
      ref={ref}
      data-step={index}
      data-key={layer.key}
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
            {said && (
              <p className="mt-2 rounded-lg border-l-2 border-accent bg-accent/5 px-2.5 py-1.5 text-[12.5px] leading-relaxed">
                <span className="font-semibold text-accent">What happened: </span>
                {said}
              </p>
            )}
            {lens && (
              <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[12px]">
                <span className="text-muted">If the model stopped here, it would say</span>
                {lens.map((guess, i) => (
                  <span key={i} className={cn("rounded px-1.5 py-0.5 font-mono", i === 0 ? "bg-accent/20 font-semibold" : "bg-surface-2 text-muted")}>
                    {ctx!.decode(guess.id)} {(guess.p * 100).toFixed(0)}%
                  </span>
                ))}
              </p>
            )}
            <div className="mt-3">{seen ? visual() : <div className="h-24 rounded-md bg-surface-2" />}</div>
            {picked !== null && (
              <div className="mt-3 rounded-lg border border-[#f7b955]/50 bg-[#f7b955]/5 p-2.5 text-[12px]" onClick={(event) => event.stopPropagation()}>
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold">How {work?.target ?? "this number"} was made</span>
                  <button type="button" onClick={() => ctx?.pick("", -1)} className="text-[11px] text-muted hover:text-ink">
                    close
                  </button>
                </div>
                {work ? (
                  <div className="mt-1.5 space-y-0.5 font-mono text-[11.5px]">
                    {work.lines.map((line, i) => (
                      <div key={i} className="flex justify-between gap-3">
                        <span className="min-w-0 truncate text-muted">{line.text}</span>
                        {line.value !== undefined && <span className={line.value >= 0 ? "text-[#f0569f]" : "text-[#5fb4ff]"}>{line.value >= 0 ? "+" : "−"}{Math.abs(line.value).toFixed(4)}</span>}
                      </div>
                    ))}
                    <div className="mt-1 flex justify-between border-t border-line pt-1 font-semibold">
                      <span>=</span>
                      <span>{work.result.toFixed(4)}</span>
                    </div>
                  </div>
                ) : (
                  <p className="mt-1 text-muted">This layer combines many weights at once (attention, a whole feed-forward network); its working is not broken down number by number. Click a number in the layer before or after it.</p>
                )}
              </div>
            )}
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
  // What the reader picked: a token to follow (the last one typed, until they choose), and a number to explain. A new input starts fresh.
  const [chosen, setChosen] = useState<number | null>(null);
  const [picked, setPicked] = useState<{ key: string; index: number } | null>(null);

  useEffect(() => {
    if (!value) return;
    let live = true;
    const timer = window.setTimeout(() => {
      void import("tessera-lab").then((lab) => {
        if (!live) return;
        try {
          setTrace({ value: runTrace(lab, value, inputs), error: null });
          setChosen(null);
          setPicked(null);
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

  const layers = useMemo(() => trace.value?.layers ?? [], [trace.value]);
  const now = layers.length ? Math.round(cursor * (layers.length - 1)) : 0;

  const used = trace.value?.used ?? 1;
  const token = Math.min(chosen ?? used - 1, used - 1);
  const byKey = useMemo(() => new Map(layers.map((layer) => [layer.key, layer])), [layers]);
  // Weights are read from the graph that made this trace, never a newer one still waiting for its own trace.
  const graph = trace.value?.graph;
  const cache = useRef(new Map<string, Record<string, { shape: number[]; data: Float32Array }>>());
  useEffect(() => cache.current.clear(), [graph]);
  const weights = useCallback(
    (key: string) => {
      if (!graph) return {};
      const known = cache.current.get(key);
      if (known) return known;
      // The graph may already be freed while its replacement is still tracing.
      let read: Record<string, { shape: number[]; data: Float32Array }> = {};
      try {
        read = graph.weights(key);
      } catch {
        return read;
      }
      cache.current.set(key, read);
      return read;
    },
    [graph],
  );
  // The logit lens: every residual-stream layer of the output's sequence, read out at the picked token.
  const lens = useMemo(() => {
    const out = new Map<string, { id: number; p: number }[]>();
    const head = [...layers].reverse().find((layer) => layer.node.op === "lmhead");
    const before = head && byKey.get(head.inputs[0]);
    const into = before && byKey.get(before.inputs[0]);
    if (!graph || !head || !into) return out;
    const width = into.shape[1];
    const stream = layers.filter((layer) => layer.rows === head.rows && layer.shape.length === 2 && layer.shape[1] === width && ["embedding", "posembed", "add"].includes(layer.node.op));
    if (!stream.length) return out;
    const row = Math.min(token, head.shape[0] - 1);
    let read: ReturnType<GraphModel["readout"]> = null;
    try {
      read = graph.readout(stream.map((layer) => layer.data.slice(row * width, (row + 1) * width)), 3);
    } catch {
      return out;
    }
    read?.forEach((guesses, i) => out.set(stream[i].key, guesses));
    return out;
  }, [layers, byKey, graph, token]);
  const decodeToken = useCallback(
    (id: number) => {
      const t = trace.value;
      if (!t) return String(id);
      if (t.decode) return t.decode(id);
      if (t.task.kind === "seq2seq") return id === 0 ? "▸" : "0123456789abcdef"[id - 1] ?? "?";
      return id === 0 ? "↵" : id === 1 ? "␣" : id >= 2 && id <= 95 ? String.fromCharCode(id + 31) : "?";
    },
    [trace.value],
  );
  const interact = useMemo<Interact>(
    () => ({ token, setToken: setChosen, picked, pick: (key, index) => setPicked(key ? { key, index } : null), byKey, weights, lens, decode: decodeToken }),
    [token, picked, byKey, weights, lens, decodeToken],
  );
  const tokens = layers.find((layer) => layer.rows && layer.node.op === "input")?.rows ?? layers.find((layer) => layer.rows)?.rows;

  useEffect(() => onSteps(layers.length), [layers.length, onSteps]);

  // Hand the values and the current layer to the 3D view beside this pane.
  const setValues = useTraceShared((state) => state.setValues);
  const setFocus = useTraceShared((state) => state.setFocus);
  useEffect(() => {
    if (!layers.length) return;
    const values = new Map<string, TracedLayer>();
    for (const layer of layers) if (layer.copy === 0) values.set(layer.nodeId, layer);
    setValues(slot, values);
  }, [layers, slot, setValues]);
  useEffect(() => {
    const layer = layers[now];
    setFocus(slot, layer ? { nodeId: layer.nodeId, copy: layer.copy, layer, rows: layer.rows } : null);
  }, [layers, now, slot, setFocus]);
  useEffect(
    () => () => {
      setValues(slot, null);
      setFocus(slot, null);
    },
    [slot, setValues, setFocus],
  );
  useEffect(() => {
    if (!playing) return;
    list.current?.querySelector(`[data-step="${now}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [now, playing]);

  if (!data) return null;
  const output = layers[layers.length - 1];
  return (
    <section className="flex min-h-0 flex-col">
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
          <InteractContext.Provider value={interact}>
          <div ref={list} className="scroll-quiet min-h-0 flex-1 space-y-0 overflow-y-auto px-3 py-3">
            {tokens && (
              <div className="sticky top-0 z-10 -mx-3 mb-3 border-b border-line bg-surface/95 px-3 py-2 backdrop-blur">
                <p className="mb-1 text-[11px] text-muted">Follow a token through every layer:</p>
                <div className="scroll-quiet flex gap-0.5 overflow-x-auto font-mono text-[12.5px]">
                  {tokens.slice(0, used).map((label, i) => (
                    <button key={i} type="button" onClick={() => setChosen(i)} className={cn("flex-none rounded px-1.5 py-0.5", i === token ? "bg-accent font-semibold text-accent-ink" : "bg-surface-2 hover:bg-surface-3")}>
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {lens.size > 0 && (
              <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3">
                <p className="text-[11px] font-medium tracking-wide text-muted uppercase">How the prediction forms</p>
                <p className="mt-0.5 mb-2 text-[11.5px] text-muted">
                  After <span className="font-mono text-ink">{tokens?.[token]}</span>, what the model would guess if it stopped at each layer. Watch the answer sharpen as the layers add their part.
                </p>
                <ol className="space-y-1">
                  {layers
                    .filter((layer) => lens.has(layer.key))
                    .map((layer) => {
                      const guesses = lens.get(layer.key)!;
                      return (
                        <li key={layer.key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] items-center gap-2 text-[11.5px]">
                          <button type="button" onClick={() => list.current?.querySelector(`[data-key="${layer.key}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" })} className="truncate text-left text-muted hover:text-ink">
                            {layer.block ? `${layer.block.replace(/ of \d+$/, "")} · ` : ""}
                            {layer.node.op === "add" ? `after ${byKey.get(layer.inputs[0])?.node.label ?? "the block"}` : (layer.node.label ?? layer.node.op)}
                          </button>
                          <span className="flex items-center gap-1.5">
                            <span className="relative h-4 flex-1 overflow-hidden rounded bg-surface-3">
                              <span className="absolute inset-y-0 left-0 rounded bg-accent/70 transition-[width] duration-500" style={{ width: `${Math.max(2, guesses[0].p * 100)}%` }} />
                              <span className="relative px-1.5 font-mono text-[11px]">{decodeToken(guesses[0].id)}</span>
                            </span>
                            <span className="w-9 text-right font-mono text-[10.5px] text-muted">{(guesses[0].p * 100).toFixed(0)}%</span>
                          </span>
                        </li>
                      );
                    })}
                </ol>
              </div>
            )}
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
          </InteractContext.Provider>
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
  const [show3d, setShow3d] = useState(true);
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
          <button type="button" onClick={() => setShow3d(!show3d)} aria-pressed={show3d} className={cn("mr-2 hidden h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[12.5px] md:flex", show3d ? "border-accent bg-accent/10 text-accent" : "border-line hover:border-muted")}>
            <Box className="size-3.5" />
            3D beside
          </button>
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
      {/* One row per model: its 3D view, painted with the trace's values, beside its trace pane. */}
      <div className={cn("grid min-h-0 flex-1", shown.length === 2 && "grid-rows-2")}>
        {shown.map((slot) => (
          <div key={slot} className={cn("flex min-h-0", slot === 1 && "border-t border-line")}>
            {show3d && (
              <div className="hidden min-w-0 flex-1 border-r border-line md:block">
                <Graph3D slot={slot} compact trace />
              </div>
            )}
            <div className={cn("flex min-h-0 min-w-0 flex-col", show3d ? "w-full md:w-[480px] md:flex-none xl:w-[560px]" : "flex-1")}>
              <Column slot={slot} inputs={inputs} cursor={cursor} playing={playing} onSteps={onSteps[slot]} onSeek={(fraction) => { setPlaying(false); setCursor(fraction); }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
