import { analyze, timesOf } from "./analyze";
import type { Analysis, ArgValue, Model, Node } from "./types";

// Comparing two architectures: headline numbers side by side, the settings
// that differ, and the blocks lined up so additions, removals and changes can
// be marked on both drawings.

export interface Metrics {
  params: number;
  activeParams: number;
  /** Compute for one example: one image, or one token for language models. */
  flops: number;
  /** Bytes to hold the weights in 16-bit precision. */
  weightBytes: number;
  /** Bytes of key/value cache per token of context, for language models. */
  kvBytesPerToken: number | null;
  /** Layers that hold weights, counting repeats. */
  weightLayers: number;
  errors: number;
}

export function metrics(model: Model, analysis: Analysis = analyze(model)): Metrics {
  let kv = 0;
  let hasAttention = false;
  let weightLayers = 0;
  for (const node of model.nodes) {
    const info = analysis.info.get(node.id);
    if (!info || info.error) continue;
    if (info.params > 0 && node.op !== "batchnorm" && node.op !== "layernorm" && node.op !== "rmsnorm") weightLayers += info.times;
    if (node.op === "attention") {
      hasAttention = true;
      const heads = Number(node.args.heads);
      const kvHeads = Number(node.args.kv ?? heads);
      const headDim = Number(node.args.head_dim ?? info.shape[1] / heads);
      kv += info.times * 2 * kvHeads * headDim * 2;
    } else if (node.op === "mla") {
      hasAttention = true;
      kv += info.times * (Number(node.args.kv_rank) + Number(node.args.qk_rope_dim)) * 2;
    }
  }
  const perExample = model.kind === "language" && model.seqLen ? analysis.flops / model.seqLen : analysis.flops;
  return {
    params: analysis.params,
    activeParams: analysis.activeParams,
    flops: perExample,
    weightBytes: analysis.params * 2,
    kvBytesPerToken: hasAttention ? kv : null,
    weightLayers,
    errors: analysis.errors.length,
  };
}

export type ChangeKind = "same" | "changed" | "added" | "removed";

/** A top-level piece of a model: a block, or a node outside any block. */
export interface Unit {
  id: string;
  label: string;
  /** What lines units up: a block's kind, or a node's op. */
  key: string;
  /** What must match for two lined-up units to count as the same. */
  signature: string;
  repeat: number;
  nodeIds: string[];
}

export interface StructuralChange {
  kind: ChangeKind;
  a?: Unit;
  b?: Unit;
  /** What differs between two lined-up units, in words. */
  detail?: string;
}

export interface Diff {
  metrics: [Metrics, Metrics];
  settings: { key: string; a: ArgValue | undefined; b: ArgValue | undefined; same: boolean }[];
  changes: StructuralChange[];
  /** For colouring the drawings: the status of every unit, by model. */
  status: [Map<string, ChangeKind>, Map<string, ChangeKind>];
}

const argsText = (node: Node) =>
  Object.entries(node.args)
    .filter(([key]) => key !== "rope")
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `${key}=${value}`)
    .join(",");

export function units(model: Model): Unit[] {
  const out: Unit[] = [];
  const topGroup = (groupId?: string): string | undefined => {
    let id = groupId;
    let last = id;
    while (id) {
      last = id;
      id = model.groups.find((group) => group.id === id)?.parent;
    }
    return last;
  };
  const seen = new Map<string, Unit>();
  for (const node of model.nodes) {
    const top = topGroup(node.group);
    if (!top) {
      out.push({ id: node.id, label: node.label ?? node.op, key: node.op, signature: `${node.op}(${argsText(node)})`, repeat: 1, nodeIds: [node.id] });
      continue;
    }
    let unit = seen.get(top);
    if (!unit) {
      const group = model.groups.find((item) => item.id === top)!;
      unit = { id: top, label: group.label, key: group.kind, signature: "", repeat: timesOf(model.groups, top), nodeIds: [] };
      seen.set(top, unit);
      out.push(unit);
    }
    unit.nodeIds.push(node.id);
  }
  // A block's signature: its operations and their settings, and how they connect.
  for (const unit of out) {
    if (unit.nodeIds.length === 1 && unit.id === unit.nodeIds[0]) continue;
    const inner = model.nodes.filter((node) => unit.nodeIds.includes(node.id));
    const index = new Map(inner.map((node, i) => [node.id, i]));
    unit.signature = `x${unit.repeat}|` + inner.map((node) => `${node.op}(${argsText(node)})<${node.inputs.map((id) => index.get(id) ?? "in").join(",")}`).join(";");
  }
  return out;
}

/** Says in a few words how two lined-up blocks differ. */
function describe(a: Unit, b: Unit, modelA: Model, modelB: Model): string {
  const parts: string[] = [];
  if (a.repeat !== b.repeat) parts.push(`repeats ${a.repeat} → ${b.repeat}`);
  const nodesA = modelA.nodes.filter((node) => a.nodeIds.includes(node.id));
  const nodesB = modelB.nodes.filter((node) => b.nodeIds.includes(node.id));
  const opsA = nodesA.map((node) => node.op);
  const opsB = nodesB.map((node) => node.op);
  for (const op of new Set([...opsA, ...opsB])) {
    const ca = opsA.filter((item) => item === op).length;
    const cb = opsB.filter((item) => item === op).length;
    if (ca !== cb) parts.push(cb > ca ? `+${cb - ca} ${op}` : `-${ca - cb} ${op}`);
  }
  // Same operations in order: report the settings that changed.
  if (!parts.length) {
    nodesA.forEach((node, i) => {
      const other = nodesB[i];
      if (!other || other.op !== node.op) return;
      for (const key of new Set([...Object.keys(node.args), ...Object.keys(other.args)])) {
        if (key === "rope") continue;
        if (String(node.args[key]) !== String(other.args[key])) parts.push(`${node.op} ${key} ${node.args[key] ?? "unset"} → ${other.args[key] ?? "unset"}`);
      }
    });
  }
  if (!parts.length) parts.push("connections differ");
  return [...new Set(parts)].slice(0, 6).join(", ");
}

export function diff(modelA: Model, modelB: Model): Diff {
  const ua = units(modelA);
  const ub = units(modelB);
  // Longest common subsequence on the unit keys, so blocks line up by kind.
  const table: number[][] = Array.from({ length: ua.length + 1 }, () => new Array(ub.length + 1).fill(0));
  for (let i = ua.length - 1; i >= 0; i--)
    for (let j = ub.length - 1; j >= 0; j--) table[i][j] = ua[i].key === ub[j].key ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
  const changes: StructuralChange[] = [];
  let i = 0;
  let j = 0;
  while (i < ua.length && j < ub.length) {
    if (ua[i].key === ub[j].key) {
      const same = ua[i].signature === ub[j].signature;
      changes.push(same ? { kind: "same", a: ua[i], b: ub[j] } : { kind: "changed", a: ua[i], b: ub[j], detail: describe(ua[i], ub[j], modelA, modelB) });
      i++;
      j++;
    } else if (table[i + 1][j] >= table[i][j + 1]) changes.push({ kind: "removed", a: ua[i++] });
    else changes.push({ kind: "added", b: ub[j++] });
  }
  while (i < ua.length) changes.push({ kind: "removed", a: ua[i++] });
  while (j < ub.length) changes.push({ kind: "added", b: ub[j++] });

  const statusA = new Map<string, ChangeKind>();
  const statusB = new Map<string, ChangeKind>();
  for (const change of changes) {
    if (change.a) statusA.set(change.a.id, change.kind);
    if (change.b) statusB.set(change.b.id, change.kind);
  }

  const keys = [...new Set([...Object.keys(modelA.meta), ...Object.keys(modelB.meta)])];
  const settings = keys.map((key) => ({ key, a: modelA.meta[key], b: modelB.meta[key], same: String(modelA.meta[key]) === String(modelB.meta[key]) }));

  return { metrics: [metrics(modelA), metrics(modelB)], settings, changes, status: [statusA, statusB] };
}

// -- Readable numbers ------------------------------------------------------------

export const formatCount = (n: number): string => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(Math.round(n));
};

export const formatBytes = (n: number): string => {
  if (n >= 1024 ** 4) return `${(n / 1024 ** 4).toFixed(2)} TB`;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
};
