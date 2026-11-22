import { OPS, type Analysis, type Category, type Group, type Model, type Node } from "tessera-engine";

// Facts about an architecture worked out from its wiring alone, with no
// training: how far gradients travel, how much of the input each output can
// see, where the compute goes and how it grows with context length.

interface Step {
  key: string;
  node: Node;
  inputs: string[];
}

const groupOf = (groups: Group[], id?: string) => (id ? groups.find((group) => group.id === id) : undefined);

/** The outermost repeated block holding `groupId`, below `scope`. */
function outermostRepeated(groups: Group[], groupId: string | undefined, scope: string | undefined) {
  let found: Group | undefined;
  for (let group = groupOf(groups, groupId); group && group.id !== scope; group = groupOf(groups, group.parent)) if ((group.repeat ?? 1) > 1) found = group;
  return found;
}

function isWithin(groups: Group[], groupId: string | undefined, target: string) {
  for (let group = groupOf(groups, groupId); group; group = groupOf(groups, group.parent)) if (group.id === target) return true;
  return false;
}

/** Every node copy in running order, with repeated blocks written out, as the trainer runs them. */
export function unroll(model: Model): Step[] {
  const steps: Step[] = [];
  const latest = new Map<string, string>();
  const emit = (nodes: Node[], scope: string | undefined, suffix: string, alias: Map<string, string>) => {
    const read = (id: string) => alias.get(id) ?? latest.get(id) ?? id;
    let i = 0;
    while (i < nodes.length) {
      const node = nodes[i];
      const repeated = outermostRepeated(model.groups, node.group, scope);
      if (!repeated) {
        const key = `${node.id}${suffix}`;
        steps.push({ key, node, inputs: node.inputs.map(read) });
        latest.set(node.id, key);
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
        if (copy > 0 && entry) copyAlias.set(entry, latest.get(exit.id)!);
        emit(members, repeated.id, `${suffix}_r${copy}`, copyAlias);
      }
    }
  };
  emit(model.nodes, undefined, "", new Map());
  return steps;
}

export interface GradientPaths {
  /** Layers with weights (norms aside) on the longest route from input to output. */
  longest: number;
  /** ... and on the shortest route, which skip connections make short. */
  shortest: number;
  /** Additions that join a path back in (residuals), counted over all copies. */
  skips: number;
  /** Concatenations that keep earlier outputs (dense and U-Net style joins). */
  joins: number;
}

export function gradientPaths(model: Model, analysis: Analysis): GradientPaths {
  const steps = unroll(model);
  const short = new Map<string, number>();
  const long = new Map<string, number>();
  let skips = 0;
  let joins = 0;
  for (const step of steps) {
    // Norms hold weights too, but papers count depth in transforming layers, so they are left out.
    const weight = (analysis.info.get(step.node.id)?.params ?? 0) > 0 && OPS[step.node.op]?.category !== "norm" ? 1 : 0;
    const ins = step.inputs.filter((key) => short.has(key));
    short.set(step.key, (ins.length ? Math.min(...ins.map((key) => short.get(key)!)) : 0) + weight);
    long.set(step.key, (ins.length ? Math.max(...ins.map((key) => long.get(key)!)) : 0) + weight);
    if (step.node.op === "add" && ins.length > 1) skips++;
    if (step.node.op === "concat" && ins.length > 1) joins++;
  }
  const last = steps[steps.length - 1]?.key;
  return { longest: long.get(last) ?? 0, shortest: short.get(last) ?? 0, skips, joins };
}

export interface FieldPoint {
  label: string;
  /** Receptive field in input pixels, and the stride (jump) between neighbouring outputs. */
  field: number;
  jump: number;
  size: number;
}

/**
 * How many input pixels each output of every spatial layer can see:
 * r = r_prev + (k - 1) · j_prev, j = j_prev · s. Branches take the widest.
 * Null for models that do not read images.
 */
export function receptiveField(model: Model, analysis: Analysis): FieldPoint[] | null {
  const input = model.nodes.find((node) => node.op === "input");
  if (!input || analysis.info.get(input.id)?.shape.length !== 3) return null;
  const steps = unroll(model);
  const state = new Map<string, { r: number; j: number }>();
  const points: FieldPoint[] = [];
  for (const step of steps) {
    const ins = step.inputs.map((key) => state.get(key)).filter((value): value is { r: number; j: number } => !!value);
    const prev = ins.length ? ins.reduce((a, b) => (b.r > a.r ? b : a)) : { r: 1, j: 1 };
    const { op, args } = step.node;
    const shape = analysis.info.get(step.node.id)?.shape ?? [];
    if (shape.length !== 3) {
      if (op === "input") state.set(step.key, prev);
      continue;
    }
    let next = prev;
    if (op === "conv2d" || op === "maxpool" || op === "avgpool") {
      const k = Number(args.k ?? (op === "conv2d" ? 3 : 2));
      const s = Number(args.s ?? (op === "conv2d" ? 1 : k));
      next = { r: prev.r + (k - 1) * prev.j, j: prev.j * s };
      points.push({ label: step.node.label ?? op, field: next.r, jump: next.j, size: shape[1] });
    } else if (op === "upconv2d") {
      next = { r: prev.r, j: prev.j / Number(args.s ?? 2) };
    }
    state.set(step.key, next);
  }
  return points;
}

/** Compute by kind of layer, counting every copy. */
export function computeByCategory(model: Model, analysis: Analysis) {
  const out = new Map<Category, number>();
  for (const node of model.nodes) {
    const info = analysis.info.get(node.id);
    if (!info?.flops) continue;
    const category = OPS[node.op]?.category ?? "reshape";
    out.set(category, (out.get(category) ?? 0) + info.flops * info.times);
  }
  return [...out.entries()].sort((a, b) => b[1] - a[1]);
}

export interface Scaling {
  /** 2 · active parameters: forward FLOPs per token outside attention's n² part. */
  linear: number;
  /** FLOPs per token per token of context: 4 · Σ heads · head_dim over attention layers. */
  quadratic: number;
  /** Bytes of key/value cache per token in 16-bit. */
  kvBytes: number;
  /** Sliding-window size if every attention layer has one. */
  window?: number;
  layers: number;
}

/** How a language model's cost grows with context length n. Null without attention. */
export function scaling(model: Model, analysis: Analysis): Scaling | null {
  let quadratic = 0;
  let kv = 0;
  let layers = 0;
  let window: number | undefined;
  let everyWindowed = true;
  for (const node of model.nodes) {
    if (node.op !== "attention" && node.op !== "mla" && node.op !== "cross_attention") continue;
    const info = analysis.info.get(node.id);
    const shape = analysis.info.get(node.inputs[0])?.shape ?? [];
    const times = info?.times ?? 1;
    const d = shape[shape.length - 1] ?? 0;
    const a = node.args;
    const heads = Number(a.heads ?? 1);
    layers += times;
    if (node.op === "mla") {
      quadratic += 2 * heads * (Number(a.qk_nope_dim ?? 0) + Number(a.qk_rope_dim ?? 0) + Number(a.v_dim ?? 0)) * times;
      kv += (Number(a.kv_rank ?? 0) + Number(a.qk_rope_dim ?? 0)) * times;
    } else {
      const hd = Number(a.head_dim ?? Math.floor(d / heads));
      quadratic += 4 * heads * hd * times;
      kv += 2 * Number(a.kv ?? heads) * hd * times;
    }
    const w = Number(a.window ?? 0);
    if (w > 0) window = Math.max(window ?? 0, w);
    else everyWindowed = false;
  }
  if (!layers) return null;
  return { linear: 2 * analysis.activeParams, quadratic, kvBytes: kv * 2, window: everyWindowed ? window : undefined, layers };
}

/** Forward FLOPs for one token when the context holds n tokens. */
export const flopsPerToken = (s: Scaling, n: number) => s.linear + s.quadratic * Math.min(n, s.window ?? n);
