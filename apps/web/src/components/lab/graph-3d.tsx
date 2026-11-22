"use client";

import { Billboard, Line, OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { BoxGeometry, CanvasTexture, CatmullRomCurve3, Color, DoubleSide, NearestFilter, Vector3, type Group } from "three";
import { formatCount, type Model, type Shape } from "tessera-engine";
import type { TracedLayer } from "tessera-lab";
import { buildDisplay, CATEGORY_COLOURS, formatShape, type Display, type DisplayNode } from "@/lib/display";
import { connect, connectionEnd } from "@/lib/edit";
import { useLab, type SlotIndex } from "@/lib/store";
import { useTraceShared } from "@/lib/trace-shared";

// The architecture as data moving through it. Each stage is the tensor a
// layer hands on, drawn the way papers draw it: an image tensor as a stack of
// feature maps, a vector as a column of neurons, a token sequence as a row of
// token vectors. Convolutions project a kernel cone from one map to the next;
// attention draws arcs between tokens; expert layers show their experts.
// Activations and norms don't change the shape, so they ride along as tags.

const MINOR = new Set(["activation", "norm", "regularise"]);
const GAP = 1.1;

const log = (n: number) => Math.log2(Math.max(2, n));

type Form = "maps" | "vector" | "tokens" | "block";

interface Stage {
  item: DisplayNode;
  form: Form;
  shape: Shape;
  /** Width along the flow, height, depth across. */
  size: [number, number, number];
  x: number;
  /** Sideways offset for side branches, such as a ResNet's downsample path. */
  z: number;
  tags: string[];
  colour: string;
}

function formOf(item: DisplayNode, language: boolean): Form {
  if (item.kind === "block") return "block";
  if (language && item.shape.length === 1) return "tokens";
  if (item.shape.length === 3) return "maps";
  if (item.shape.length === 2) return "tokens";
  return "vector";
}

function sizeOf(form: Form, shape: Shape): [number, number, number] {
  if (form === "maps" || (form === "block" && shape.length === 3)) {
    const side = 0.7 + log(shape[1]) * 0.42;
    return [0.25 + log(shape[0]) * 0.16, side, side];
  }
  if (form === "tokens" || (form === "block" && shape.length === 2)) return [0.5, shape.length === 1 ? 0.35 : 0.8 + log(shape[1]) * 0.22, 2.6];
  if (form === "vector") return [0.3, Math.min(4, 0.6 + log(shape[0]) * 0.32), 0.3];
  return [1, 1.4, 1.4];
}

/** A faint grid, so a feature map reads as an image of values rather than a slab. */
function gridTexture(colour: string, cells: number) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colour;
  ctx.globalAlpha = 0.85;
  ctx.fillRect(0, 0, 64, 64);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(11,13,18,0.55)";
  ctx.lineWidth = 1;
  const step = 64 / cells;
  for (let i = 0; i <= cells; i++) {
    ctx.beginPath();
    ctx.moveTo(i * step, 0);
    ctx.lineTo(i * step, 64);
    ctx.moveTo(0, i * step);
    ctx.lineTo(64, i * step);
    ctx.stroke();
  }
  const texture = new CanvasTexture(canvas);
  texture.magFilter = NearestFilter;
  return texture;
}

// -- Real values, from the trace -----------------------------------------------------

const BG = [18, 21, 28];
const POS = [240, 86, 159];
const NEG = [95, 180, 255];

/** Paints values onto a canvas texture: pink for positive, blue for negative, dark near zero. */
function paintTexture(width: number, height: number, at: (x: number, y: number) => number, scale: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const image = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const t = Math.max(-1, Math.min(1, at(x, y) / (scale || 1)));
      const colour = t >= 0 ? POS : NEG;
      const a = Math.pow(Math.abs(t), 0.7);
      const o = (y * width + x) * 4;
      for (let k = 0; k < 3; k++) image.data[o + k] = BG[k] + (colour[k] - BG[k]) * a;
      image.data[o + 3] = 255;
    }
  ctx.putImageData(image, 0, 0);
  const texture = new CanvasTexture(canvas);
  texture.magFilter = NearestFilter;
  return texture;
}

const maxAbs = (data: Float32Array) => {
  let m = 0;
  for (let i = 0; i < data.length; i++) m = Math.max(m, Math.abs(data[i]));
  return m;
};

/** One channel of a [H, W, C] layer (tf's layout), or how strongly all channels fire when channel is -1. */
function mapTexture(layer: TracedLayer, channel: number) {
  const [h, w, c] = layer.shape;
  const d = layer.data;
  if (channel >= 0) return paintTexture(w, h, (x, y) => d[(y * w + x) * c + channel], maxAbs(d));
  const energy = new Float32Array(h * w);
  for (let p = 0; p < h * w; p++) {
    let sum = 0;
    for (let k = 0; k < c; k++) sum += Math.abs(d[p * c + k]);
    energy[p] = sum / c;
  }
  return paintTexture(w, h, (x, y) => energy[y * w + x], maxAbs(energy));
}

/** A picture of any layer's output for the floating panel: a mosaic of channels, a token-by-feature grid, or a vector folded square. */
function panelTexture(layer: TracedLayer) {
  const { shape, data } = layer;
  if (layer.attention) {
    const [, q, k] = layer.attention.shape;
    return paintTexture(k, q, (x, y) => layer.attention!.data[y * k + x], 1);
  }
  if (shape.length === 3) {
    const [h, w, c] = shape;
    const n = Math.min(16, c);
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const scale = maxAbs(data);
    return paintTexture(cols * (w + 1), rows * (h + 1), (x, y) => {
      const k = Math.floor(y / (h + 1)) * cols + Math.floor(x / (w + 1));
      const [px, py] = [x % (w + 1), y % (h + 1)];
      return k >= n || px === w || py === h ? 0 : data[(py * w + px) * c + k];
    }, scale);
  }
  if (shape.length === 2) {
    const [rows, d] = shape;
    const cols = Math.min(d, 128);
    return paintTexture(cols, Math.min(rows, 64), (x, y) => data[y * d + x], maxAbs(data));
  }
  const n = Math.min(data.length, 1024);
  const side = Math.ceil(Math.sqrt(n));
  return paintTexture(side, side, (x, y) => (y * side + x < n ? data[y * side + x] : 0), maxAbs(data.subarray(0, n)));
}

/** The values of the layer the trace is on, floating above its stage. */
function ValuePanel({ layer, height, label }: { layer: TracedLayer; height: number; label: string }) {
  const texture = useMemo(() => panelTexture(layer), [layer]);
  const image = texture.image as HTMLCanvasElement;
  const aspect = image.width / image.height;
  const h = Math.min(1.8, 3 / Math.max(aspect, 0.6));
  const w = h * aspect;
  return (
    <Billboard position={[0, height + 1.1 + h / 2, 0]}>
      <mesh position={[0, 0, -0.01]}>
        <planeGeometry args={[w + 0.16, h + 0.16]} />
        <meshBasicMaterial color="#f0569f" />
      </mesh>
      <mesh>
        <planeGeometry args={[w, h]} />
        <meshBasicMaterial map={texture} />
      </mesh>
      <Text position={[0, h / 2 + 0.2, 0]} fontSize={0.2} color="#eef0f5" anchorX="center" anchorY="bottom" outlineWidth={0.012} outlineColor="#0b0d12">
        {label}
      </Text>
    </Billboard>
  );
}

function Maps({ stage, lit, value }: { stage: Stage; lit: number; value?: TracedLayer }) {
  const [depth, side] = [stage.size[0], stage.size[1]];
  const slices = Math.max(2, Math.min(14, Math.round(log(stage.shape[0]) * 1.6)));
  const grid = useMemo(() => gridTexture(stage.colour, Math.max(2, Math.min(12, Math.round(log(stage.shape[1]) * 1.6)))), [stage.colour, stage.shape]);
  // With a trace, the slices show real channels and the front one how strongly the whole layer fires.
  const real = useMemo(() => (value && value.shape.length === 3 ? Array.from({ length: slices }, (_, i) => mapTexture(value, i === slices - 1 ? -1 : Math.floor((i / slices) * value.shape[2]))) : null), [value, slices]);
  return (
    <group>
      {Array.from({ length: slices }, (_, i) => (
        <mesh key={i} position={[-depth / 2 + (i / Math.max(1, slices - 1)) * depth, side / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[side, side]} />
          {real ? (
            <meshBasicMaterial map={real[i]} transparent opacity={i === slices - 1 ? 1 : 0.55 + 0.3 * lit} side={DoubleSide} depthWrite={false} />
          ) : (
            <meshStandardMaterial map={grid} transparent opacity={0.28 + 0.5 * lit + (i === slices - 1 ? 0.25 : 0)} side={DoubleSide} emissive={new Color(stage.colour)} emissiveIntensity={0.15 + lit * 0.9} depthWrite={false} />
          )}
        </mesh>
      ))}
    </group>
  );
}

function Neurons({ stage, lit }: { stage: Stage; lit: number }) {
  const count = Math.min(stage.shape[0] ?? 1, 18);
  const height = stage.size[1];
  const more = (stage.shape[0] ?? 1) > count;
  return (
    <group>
      {Array.from({ length: count }, (_, i) => {
        const y = 0.15 + (i / Math.max(1, count - 1)) * (height - 0.3);
        if (more && i === Math.floor(count / 2)) return null;
        return (
          <mesh key={i} position={[0, y, 0]}>
            <sphereGeometry args={[0.075, 14, 14]} />
            <meshStandardMaterial color={stage.colour} emissive={new Color(stage.colour)} emissiveIntensity={0.25 + lit * 1.2} roughness={0.35} />
          </mesh>
        );
      })}
      {more && (
        <Text position={[0, height / 2, 0]} fontSize={0.22} color="#9aa2b4" anchorX="center" anchorY="middle">
          ⋮
        </Text>
      )}
    </group>
  );
}

const TOKENS = 7;
function Tokens({ stage, lit, attention, experts, topk }: { stage: Stage; lit: number; attention: boolean; experts: number; topk: number }) {
  const [width, height, span] = stage.size;
  const zOf = (i: number) => -span / 2 + (i / (TOKENS - 1)) * span;
  const arcs = useMemo(() => {
    if (!attention) return [];
    const out: { points: Vector3[]; colour: string }[] = [];
    const heads = ["#fb7185", "#f0569f", "#f7b955", "#5fb4ff"];
    for (let q = 1; q < TOKENS; q++) {
      for (let k = 0; k < q; k += Math.max(1, Math.floor(q / 2))) {
        const a = new Vector3(0, height + 0.05, zOf(k));
        const b = new Vector3(0, height + 0.05, zOf(q));
        const mid = a.clone().lerp(b, 0.5).add(new Vector3(0, 0.25 + (q - k) * 0.12, 0));
        out.push({ points: new CatmullRomCurve3([a, mid, b]).getPoints(16), colour: heads[(q + k) % heads.length] });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attention, height, span]);
  return (
    <group>
      {Array.from({ length: TOKENS }, (_, i) => (
        <mesh key={i} position={[0, height / 2, zOf(i)]}>
          <boxGeometry args={[width, height, 0.16]} />
          <meshStandardMaterial color={stage.colour} transparent opacity={0.55 + lit * 0.35} emissive={new Color(stage.colour)} emissiveIntensity={0.15 + lit * 0.8} />
        </mesh>
      ))}
      {arcs.map((arc, i) => (
        <Line key={i} points={arc.points} color={arc.colour} lineWidth={1.4} transparent opacity={0.85} />
      ))}
      {experts > 0 &&
        Array.from({ length: Math.min(experts, 8) }, (_, e) => (
          <mesh key={e} position={[width / 2 + 0.35, 0.2 + e * ((height - 0.2) / 8), 0]}>
            <boxGeometry args={[0.22, 0.18, 0.22]} />
            <meshStandardMaterial color={e < topk ? "#c084fc" : "#2a2f3c"} emissive={new Color(e < topk ? "#c084fc" : "#000")} emissiveIntensity={e < topk ? 0.6 + lit : 0} />
          </mesh>
        ))}
    </group>
  );
}

/** A closed block draws the tensor it hands on, inside a thin outline, with a dashed arc when it holds a residual connection. */
function BlockStage({ stage, lit, residual }: { stage: Stage; lit: number; residual: boolean }) {
  const [w, h, d] = stage.size;
  const outline = useMemo(() => new BoxGeometry(w + 0.3, h + 0.3, d + 0.3), [w, h, d]);
  const tinted = { ...stage, colour: "#f6782a" };
  const arc = useMemo(() => new CatmullRomCurve3([new Vector3(-w / 2 - 0.15, h * 0.75, 0), new Vector3(0, h + 0.9, 0), new Vector3(w / 2 + 0.15, h * 0.75, 0)]).getPoints(20), [w, h]);
  return (
    <group>
      {stage.shape.length === 3 ? <Maps stage={tinted} lit={lit} /> : stage.shape.length === 2 ? <Tokens stage={tinted} lit={lit} attention={false} experts={0} topk={0} /> : <Neurons stage={tinted} lit={lit} />}
      <lineSegments position={[0, h / 2, 0]}>
        <edgesGeometry args={[outline]} />
        <lineBasicMaterial color="#f6782a" transparent opacity={0.55} />
      </lineSegments>
      {residual && <Line points={arc} color="#f0569f" lineWidth={2} dashed dashSize={0.16} gapSize={0.1} />}
    </group>
  );
}

function inGroup(model: Model, groupId: string | undefined, target: string): boolean {
  let id = groupId;
  while (id) {
    if (id === target) return true;
    id = model.groups.find((group) => group.id === id)?.parent;
  }
  return false;
}

/** One stage: the tensor, its labels, and how bright the forward pass has made it. */
function StageView({ stage, index, model, selected, onSelect, pass, value, focus }: { stage: Stage; index: number; model: Model; selected: boolean; onSelect: (shift: boolean) => void; pass: React.RefObject<number>; value?: TracedLayer; focus?: { layer: TracedLayer; label: string } }) {
  const group = useRef<Group>(null);
  const [lit, setLit] = useState(0);
  useFrame(() => {
    // The pulse of the forward pass brightens each stage as it passes.
    const distance = Math.abs((pass.current ?? 0) - index);
    const next = Math.max(0, 1 - distance * 0.9);
    if (Math.abs(next - lit) > 0.04) setLit(next);
  });
  const node = model.nodes.find((item) => item.id === stage.item.id);
  const attention = node?.op === "attention" || node?.op === "mla";
  const experts = node?.op === "moe" ? Number(node.args.experts) : 0;
  // A block holds a residual when one of its joins reads straight from outside the block.
  const residual =
    stage.form === "block" &&
    model.nodes.some((item) => {
      if (item.op !== "add" || !inGroup(model, item.group, stage.item.id)) return false;
      return item.inputs.some((id) => {
        const source = model.nodes.find((other) => other.id === id);
        return !source || !inGroup(model, source.group, stage.item.id);
      });
    });
  const topk = node?.op === "moe" ? Number(node.args.topk ?? 2) : 0;
  const height = stage.size[1];
  const title = stage.item.label + (stage.item.repeat > 1 ? `  ×${stage.item.repeat}` : "");
  return (
    <group
      ref={group}
      position={[stage.x, 0, stage.z]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(event.shiftKey);
      }}
    >
      {stage.form === "maps" && <Maps stage={stage} lit={lit} value={value} />}
      {stage.form === "vector" && <Neurons stage={stage} lit={lit} />}
      {stage.form === "tokens" && <Tokens stage={stage} lit={lit} attention={attention} experts={experts} topk={topk} />}
      {stage.form === "block" && <BlockStage stage={stage} lit={lit} residual={residual} />}
      {(selected || focus) && (
        <mesh position={[0, height / 2, 0]}>
          <boxGeometry args={[stage.size[0] + 0.25, height + 0.25, stage.size[2] + 0.25]} />
          <meshBasicMaterial color="#f0569f" wireframe />
        </mesh>
      )}
      {focus && <ValuePanel layer={focus.layer} height={height + (attention ? 1 : 0)} label={focus.label} />}
      {/* Neighbouring labels alternate heights so long names do not run into each other. */}
      <Billboard position={[0, height + (attention ? 1.4 : 0.45) + (index % 2) * 0.55, 0]}>
        <Text fontSize={0.26} color="#eef0f5" anchorX="center" anchorY="bottom" outlineWidth={0.015} outlineColor="#0b0d12" maxWidth={4}>
          {title}
        </Text>
        <Text position={[0, -0.05, 0]} fontSize={0.17} color={stage.colour} anchorX="center" anchorY="top" outlineWidth={0.01} outlineColor="#0b0d12">
          {formatShape(stage.shape, model)}
          {stage.item.params ? `  ·  ${formatCount(stage.item.params)}` : ""}
        </Text>
      </Billboard>
      {stage.tags.length > 0 && (
        <Billboard position={[0, -0.32, 0]}>
          <Text fontSize={0.15} color="#9aa2b4" anchorX="center" anchorY="top" outlineWidth={0.008} outlineColor="#0b0d12">
            {stage.tags.join(" · ")}
          </Text>
        </Billboard>
      )}
    </group>
  );
}

/** The cone a convolution's kernel sweeps: a k×k patch on the input map to one point on the output. */
function KernelCone({ from, to, k, inSide }: { from: Stage; to: Stage; k: number; inSide: number }) {
  const patch = Math.max(0.12, Math.min(from.size[1] * 0.5, (k / Math.max(1, inSide)) * from.size[1] * 2.2));
  const fx = from.x + from.size[0] / 2;
  const tx = to.x - to.size[0] / 2;
  const fy = from.size[1] * 0.62;
  const ty = to.size[1] * 0.55;
  const corners = [
    [fy + patch / 2, from.z + patch / 2],
    [fy + patch / 2, from.z - patch / 2],
    [fy - patch / 2, from.z - patch / 2],
    [fy - patch / 2, from.z + patch / 2],
  ];
  return (
    <group>
      <Line points={[...corners, corners[0]].map(([y, z]) => [fx, y, z] as [number, number, number])} color="#f0569f" lineWidth={1.6} />
      {corners.map(([y, z], i) => (
        <Line key={i} points={[[fx, y, z], [tx, ty, to.z]]} color="#f0569f" lineWidth={1} transparent opacity={0.7} />
      ))}
    </group>
  );
}

/** A fan of connections between two neuron columns: every input reaches every output. */
function DenseFan({ from, to }: { from: Stage; to: Stage }) {
  const lines = useMemo(() => {
    const out: [number, number, number][][] = [];
    const a = 6;
    const b = 6;
    for (let i = 0; i < a; i++)
      for (let j = 0; j < b; j++)
        out.push([
          [from.x, 0.2 + (i / (a - 1)) * (from.size[1] - 0.4), from.z],
          [to.x, 0.2 + (j / (b - 1)) * (to.size[1] - 0.4), to.z],
        ]);
    return out;
  }, [from, to]);
  return (
    <group>
      {lines.map((points, i) => (
        <Line key={i} points={points} color="#a78bfa" lineWidth={0.6} transparent opacity={0.22} />
      ))}
    </group>
  );
}

/** Frames the whole pipeline once it is laid out: from the front and to one side, so depth and token arcs both read. */
function FitCamera({ length, closer = false }: { length: number; closer?: boolean }) {
  const get = useThree((state) => state.get);
  useEffect(() => {
    const { camera, controls } = get() as unknown as { camera: import("three").PerspectiveCamera; controls: { target: Vector3; update: () => void } | null };
    const span = Math.max(8, length);
    const distance = (span / 2 / Math.tan((camera.fov * Math.PI) / 360)) * (closer ? 0.6 : 0.82);
    camera.position.set(length / 2 - distance * 0.16, distance * 0.38, distance * 0.9);
    camera.far = distance * 8 + 100;
    camera.updateProjectionMatrix();
    if (controls) {
      controls.target.set(length / 2, 1, 0);
      controls.update();
    } else camera.lookAt(length / 2, 1, 0);
  }, [get, length, closer]);
  return null;
}

function Scene({ display, slot, model, compact, trace }: { display: Display; slot: SlotIndex; model: Model; compact: boolean; trace: boolean }) {
  const selection = useLab((state) => state.selection);
  const focus = useTraceShared((state) => (trace ? state.focus[slot] : null));
  const values = useTraceShared((state) => (trace ? state.values[slot] : null));
  const follow = useRef<number | null>(null);
  const select = useLab((state) => state.select);
  const edit = useLab((state) => state.edit);
  const pass = useRef(0);
  // Shift-click wires the selected layer into the clicked one, as dragging between dots does in 2D.
  const pick = (id: string, shift: boolean) => {
    if (shift && selection?.slot === slot && selection.id !== id) {
      const from = selection.id;
      edit(slot, (current) => {
        const source = connectionEnd(current, from, "source");
        const target = connectionEnd(current, id, "target");
        return source && target ? connect(current, source, target) : current;
      });
      return;
    }
    select({ slot, id });
  };
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const { stages, links, frames, length } = useMemo(() => {
    const ordered = display.nodes.filter((item) => !item.open).sort((a, b) => a.ay - b.ay || a.depth - b.depth);
    const stages: Stage[] = [];
    const owner = new Map<string, Stage>();
    const inputOf = new Map<string, string>();
    for (const edge of display.edges) if (!inputOf.has(edge.target)) inputOf.set(edge.target, edge.source);
    // Sideways position from the 2D layout keeps parallel branches (an encoder beside a decoder, a ResNet shortcut) side by side.
    const xs = ordered.map((item) => item.ax + item.width / 2);
    const centre = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
    // Shape-keeping layers (activations, norms) ride on the stage they read from.
    const major: DisplayNode[] = [];
    for (const item of ordered) {
      const source = inputOf.get(item.id);
      const host = source ? owner.get(source) : undefined;
      if (MINOR.has(item.category) && host && item.shape.join() === host.shape.join()) {
        host.tags.push(item.op ?? item.label);
        owner.set(item.id, host);
        continue;
      }
      major.push(item);
      const form = formOf(item, model.kind === "language");
      const stage: Stage = { item, form, shape: item.shape, size: sizeOf(form, item.shape), x: 0, z: ((item.ax + item.width / 2 - centre) / 240) * 3.8, tags: [], colour: CATEGORY_COLOURS[item.category] };
      stages.push(stage);
      owner.set(item.id, stage);
    }
    // Stages at the same level of the layout share a position along the flow.
    const levels: Stage[][] = [];
    for (const stage of stages) {
      const level = levels.find((row) => Math.abs(row[0].item.ay - stage.item.ay) < 12);
      if (level) level.push(stage);
      else levels.push([stage]);
    }
    levels.sort((a, b) => a[0].item.ay - b[0].item.ay);
    let x = 0;
    let previousWidth = 0;
    for (const level of levels) {
      const width = Math.max(...level.map((stage) => stage.size[0]));
      x += previousWidth / 2 + GAP + width / 2;
      for (const stage of level) stage.x = x;
      previousWidth = width;
    }
    stages.sort((a, b) => a.x - b.x);
    void major;
    const links: { from: Stage; to: Stage; skip: boolean }[] = [];
    const seen = new Set<string>();
    for (const edge of display.edges) {
      const from = owner.get(edge.source);
      const to = owner.get(edge.target);
      if (!from || !to || from === to) continue;
      const key = `${stages.indexOf(from)}-${stages.indexOf(to)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      links.push({ from, to, skip: stages.indexOf(to) - stages.indexOf(from) > 1 });
    }
    // Open blocks: a frame around their stages, labelled with how often they repeat.
    const frames = display.nodes
      .filter((item) => item.open)
      .map((item) => {
        const inside = stages.filter((stage) => {
          let parent = stage.item.parent;
          while (parent) {
            if (parent === item.id) return true;
            parent = display.nodes.find((other) => other.id === parent)?.parent;
          }
          return false;
        });
        if (!inside.length) return null;
        const start = Math.min(...inside.map((stage) => stage.x - stage.size[0] / 2)) - 0.45;
        const end = Math.max(...inside.map((stage) => stage.x + stage.size[0] / 2)) + 0.45;
        const height = Math.max(...inside.map((stage) => stage.size[1])) + 0.6;
        const depth = Math.max(...inside.map((stage) => stage.size[2] + Math.abs(stage.z) * 2)) + 0.8;
        return { item, start, end, height, depth };
      })
      .filter((frame): frame is NonNullable<typeof frame> => !!frame);
    return { stages, links, frames, length: x + 2 };
  }, [display, model.kind]);

  // Which stage shows the traced layer: its own, the block holding it, or the stage it rides on as a tag.
  const focusIndex = useMemo(() => {
    if (!focus) return -1;
    const node = model.nodes.find((item) => item.id === focus.nodeId);
    const direct = stages.findIndex((stage) => stage.item.id === focus.nodeId || (stage.item.kind === "block" && inGroup(model, node?.group, stage.item.id)));
    if (direct >= 0) return direct;
    // Activations and norms ride on the stage of the layer they read from.
    let source = node?.inputs[0];
    while (source) {
      const index = stages.findIndex((stage) => stage.item.id === source);
      if (index >= 0) return index;
      source = model.nodes.find((item) => item.id === source)?.inputs[0];
    }
    return -1;
  }, [focus, stages, model]);

  useEffect(() => {
    follow.current = focusIndex >= 0 ? stages[focusIndex].x : null;
  }, [focusIndex, stages]);

  useFrame((state, delta) => {
    if (trace) {
      // The pulse sits on the traced layer, and the camera glides along to it.
      if (focusIndex >= 0) pass.current += (focusIndex - pass.current) * Math.min(1, delta * 6);
      const controls = state.controls as unknown as { target: Vector3; update: () => void } | null;
      if (controls && follow.current !== null) {
        const dx = (follow.current - controls.target.x) * Math.min(1, delta * 3);
        controls.target.x += dx;
        state.camera.position.x += dx;
        controls.update();
        if (Math.abs(follow.current - controls.target.x) < 0.01) follow.current = null;
      }
      return;
    }
    if (!reduced) pass.current = (pass.current + delta * 1.6) % (stages.length + 3);
  });

  return (
    <>
      <color attach="background" args={["#0b0d12"]} />
      <hemisphereLight args={["#cbd6ff", "#1a1d26", 0.8]} />
      <directionalLight position={[length * 0.3, 12, 8]} intensity={1.3} />
      <ambientLight intensity={0.35} />
      <gridHelper args={[Math.max(20, length * 1.4), 50, "#232838", "#151922"]} position={[length / 2, -0.4, 0]} />
      {links.map(({ from, to, skip }, i) => {
        const node = model.nodes.find((item) => item.id === to.item.id);
        if (!skip && to.form === "maps" && from.form === "maps" && node?.op === "conv2d") return <KernelCone key={i} from={from} to={to} k={Number(node.args.k ?? 3)} inSide={from.shape[1]} />;
        if (!skip && to.form === "vector" && from.form === "vector") return <DenseFan key={i} from={from} to={to} />;
        const a = new Vector3(from.x + from.size[0] / 2, from.size[1] / 2, from.z);
        const b = new Vector3(to.x - to.size[0] / 2, to.size[1] / 2, to.z);
        // Skip connections arch over the stages they jump.
        const mid = a.clone().lerp(b, 0.5).add(new Vector3(0, skip ? 1.2 + a.distanceTo(b) * 0.12 : 0, 0));
        return <Line key={i} points={new CatmullRomCurve3([a, mid, b]).getPoints(24)} color={skip ? "#f0569f" : "#56607a"} lineWidth={skip ? 2.4 : 1.2} dashed={skip} dashSize={0.2} gapSize={0.12} />;
      })}
      {frames.map((frame) => (
        <group key={frame.item.id} position={[(frame.start + frame.end) / 2, 0, 0]}>
          <mesh position={[0, frame.height / 2 - 0.3, 0]}>
            <boxGeometry args={[frame.end - frame.start, frame.height, frame.depth]} />
            <meshBasicMaterial color="#f6782a" transparent opacity={0.05} depthWrite={false} />
          </mesh>
          <mesh position={[0, frame.height / 2 - 0.3, 0]}>
            <boxGeometry args={[frame.end - frame.start, frame.height, frame.depth]} />
            <meshBasicMaterial color="#f6782a" wireframe transparent opacity={0.3} />
          </mesh>
          <Billboard position={[0, frame.height + 1.6, 0]}>
            <Text fontSize={0.34} color="#f6782a" anchorX="center" anchorY="bottom" outlineWidth={0.015} outlineColor="#0b0d12">
              {frame.item.label}
              {frame.item.repeat > 1 ? `  ×${frame.item.repeat}` : ""}
            </Text>
            {frame.item.repeat > 1 && (
              <Text position={[0, -0.06, 0]} fontSize={0.17} color="#9aa2b4" anchorX="center" anchorY="top" outlineWidth={0.01} outlineColor="#0b0d12">
                this block runs {frame.item.repeat} times in a row
              </Text>
            )}
          </Billboard>
        </group>
      ))}
      <FitCamera length={length} closer={compact} />
      {stages.map((stage, index) => (
        <StageView
          key={stage.item.id}
          stage={stage}
          index={index}
          model={model}
          pass={pass}
          selected={selection?.slot === slot && selection.id === stage.item.id}
          onSelect={(shift) => pick(stage.item.id, shift)}
          value={values?.get(stage.item.id)}
          focus={index === focusIndex && focus ? { layer: focus.layer, label: `${model.nodes.find((item) => item.id === focus.nodeId)?.label ?? focus.layer.nodeId}${focus.copy ? ` (copy ${focus.copy + 1})` : ""}` } : undefined}
        />
      ))}
      <OrbitControls makeDefault enableDamping target={[length / 2, 1, 0]} maxPolarAngle={Math.PI / 1.9} />
    </>
  );
}

export default function Graph3D({ slot, compact = false, trace = false }: { slot: SlotIndex; compact?: boolean; trace?: boolean }) {
  const data = useLab((state) => state.slots[slot]);
  const status = useLab((state) => state.diff?.status[slot]);
  const select = useLab((state) => state.select);
  const [display, setDisplay] = useState<Display | null>(null);

  useEffect(() => {
    if (!data) return;
    let live = true;
    void buildDisplay(data.model, data.analysis, data.expanded, status).then((result) => {
      if (live) setDisplay(result);
    });
    return () => {
      live = false;
    };
  }, [data, status]);

  if (!data || !display) return <div className="grid h-full place-items-center text-[13px] text-muted">Laying out…</div>;
  return (
    <div className="relative h-full w-full">
      <Canvas dpr={[1, 2]} camera={{ position: [0, 6, 14], fov: 42, near: 0.05, far: 400 }} onPointerMissed={() => select(null)}>
        <Scene display={display} slot={slot} model={data.model} compact={compact} trace={trace} />
      </Canvas>
      {!compact && !trace && <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-line bg-surface/85 px-3 py-2 text-[11.5px] leading-relaxed text-muted backdrop-blur">
        <p>
          <span className="text-ink">Feature maps</span> stack by channels · <span className="text-[#f0569f]">cones</span> are convolution kernels · <span className="text-ink">dots</span> are neurons
        </p>
        <p>
          <span className="text-ink">Bars</span> are token vectors, arcs are attention · <span className="text-[#f0569f]">dashed arcs</span> are skip connections · drag to orbit, scroll to zoom
        </p>
        <p>
          Click a layer, then <span className="text-ink">shift-click</span> another to connect the first into it
        </p>
      </div>}
    </div>
  );
}
