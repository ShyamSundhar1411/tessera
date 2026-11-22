import { analyze, OPS, type ArgValue, type Model, type Node } from "tessera-engine";

// Edits made on the canvas or in the inspector. Each returns a new model and
// keeps the nodes in an order where every node comes after its inputs, which
// the analysis and the text format rely on.

export const DEFAULT_ARGS: Record<string, Record<string, ArgValue>> = {
  conv2d: { out: 32, k: 3 },
  linear: { out: 64 },
  maxpool: { k: 2 },
  avgpool: { k: 2 },
  dropout: { rate: 0.1 },
  embedding: { vocab: 1000, dim: 64 },
  attention: { heads: 4 },
  mla: { heads: 4, kv_rank: 32, qk_nope_dim: 16, qk_rope_dim: 8, v_dim: 16 },
  mlp: { ff: 256, act: "swiglu" },
  moe: { experts: 8, topk: 2, ff: 128 },
  lmhead: { vocab: 1000 },
  concat: {},
};

/** Puts nodes in an order where inputs come first, keeping the existing order wherever it already works. */
export function topoSort(model: Model): Model {
  const byId = new Map(model.nodes.map((node) => [node.id, node]));
  const placed = new Set<string>();
  const out: Node[] = [];
  const visiting = new Set<string>();
  const visit = (node: Node) => {
    if (placed.has(node.id) || visiting.has(node.id)) return;
    visiting.add(node.id);
    for (const id of node.inputs) {
      const input = byId.get(id);
      if (input) visit(input);
    }
    visiting.delete(node.id);
    placed.add(node.id);
    out.push(node);
  };
  model.nodes.forEach(visit);
  return { ...model, nodes: out };
}

const freshId = (model: Model, op: string) => {
  let n = 1;
  while (model.nodes.some((node) => node.id === (n === 1 ? op : `${op}_${n}`))) n++;
  return n === 1 ? op : `${op}_${n}`;
};

export function addNode(model: Model, op: string, after?: string, group?: string): { model: Model; id: string } {
  const id = freshId(model, op);
  const node: Node = { id, op, args: { ...(DEFAULT_ARGS[op] ?? {}) }, inputs: after ? [after] : [], ...(group ? { group } : {}) };
  return { model: topoSort({ ...model, nodes: [...model.nodes, node] }), id };
}

/**
 * Connects source's output to target. A join (add, concat) gains an input; a
 * layer that reads nothing yet reads source. A layer that already reads
 * something keeps it and gets a join in front: an add when the shapes match,
 * as a residual, or a concat when only the channels differ, as a dense
 * connection. So dragging an earlier layer onto a later one draws a skip.
 */
export function connect(model: Model, source: string, target: string): Model {
  if (source === target) return model;
  // Refuse a connection that would make a loop.
  const reaches = (from: string, to: string, seen = new Set<string>()): boolean => {
    if (from === to) return true;
    if (seen.has(from)) return false;
    seen.add(from);
    return model.nodes.filter((node) => node.inputs.includes(from)).some((node) => reaches(node.id, to, seen));
  };
  if (reaches(target, source)) return model;
  const node = model.nodes.find((item) => item.id === target);
  const arity = node && OPS[node.op]?.arity;
  if (!node || arity === "none" || node.inputs.includes(source)) return model;
  if (arity === "many" || node.inputs.length === 0) {
    const nodes = model.nodes.map((item) => (item.id === target ? { ...item, inputs: [...item.inputs, source] } : item));
    return topoSort({ ...model, nodes });
  }
  const info = analyze(model).info;
  const a = info.get(node.inputs[0])?.shape ?? [];
  const b = info.get(source)?.shape ?? [];
  const same = a.join("x") === b.join("x");
  // Channels first for images, features last for sequences: everything else must match to concatenate.
  const concatenable = a.length === b.length && (a.length === 3 ? a.slice(1).join("x") === b.slice(1).join("x") : a.slice(0, -1).join("x") === b.slice(0, -1).join("x"));
  const op = same || !concatenable ? "add" : "concat";
  const id = freshId(model, op);
  const join: Node = { id, op, args: {}, inputs: [node.inputs[0], source], label: op === "add" ? "Skip join" : "Dense join", ...(node.group ? { group: node.group } : {}) };
  const nodes = [...model.nodes.map((item) => (item.id === target ? { ...item, inputs: [id, ...item.inputs.slice(1)] } : item)), join];
  return topoSort({ ...model, nodes });
}

/**
 * The real node a canvas box stands for at one end of a connection: itself,
 * or for a closed block, the node its output leaves from (as a source) or the
 * first node that reads from outside it (as a target).
 */
export function connectionEnd(model: Model, id: string, end: "source" | "target"): string | undefined {
  if (model.nodes.some((node) => node.id === id)) return id;
  const within = (groupId: string | undefined): boolean => {
    for (let g = groupId; g; g = model.groups.find((group) => group.id === g)?.parent) if (g === id) return true;
    return false;
  };
  const members = model.nodes.filter((node) => within(node.group));
  if (!members.length) return undefined;
  if (end === "source") return members[members.length - 1].id;
  const inside = new Set(members.map((node) => node.id));
  return members.find((node) => node.inputs.some((input) => !inside.has(input)))?.id ?? members[0].id;
}

export function disconnect(model: Model, source: string, target: string): Model {
  return { ...model, nodes: model.nodes.map((node) => (node.id === target ? { ...node, inputs: node.inputs.filter((id) => id !== source) } : node)) };
}

/**
 * Removes a node. When it had one input, its readers are reconnected to that
 * input, so deleting a layer from the middle of a chain keeps the chain whole.
 */
export function removeNode(model: Model, id: string): Model {
  const node = model.nodes.find((item) => item.id === id);
  if (!node || node.op === "input") return model;
  const bridge = node.inputs.length === 1 ? node.inputs[0] : null;
  const nodes = model.nodes
    .filter((item) => item.id !== id)
    .map((item) =>
      item.inputs.includes(id) ? { ...item, inputs: [...new Set(item.inputs.flatMap((input) => (input === id ? (bridge ? [bridge] : []) : [input])))] } : item,
    );
  return { ...model, nodes };
}

export function updateNode(model: Model, id: string, change: { args?: Record<string, ArgValue | undefined>; label?: string; op?: string }): Model {
  return {
    ...model,
    nodes: model.nodes.map((node) => {
      if (node.id !== id) return node;
      const args = { ...node.args };
      for (const [key, value] of Object.entries(change.args ?? {})) {
        if (value === undefined || value === "") delete args[key];
        else args[key] = value;
      }
      return { ...node, args, ...(change.label !== undefined ? { label: change.label || undefined } : {}), ...(change.op ? { op: change.op } : {}) };
    }),
  };
}

/** Changes how many times a block repeats. */
export function setRepeat(model: Model, groupId: string, repeat: number): Model {
  return { ...model, groups: model.groups.map((group) => (group.id === groupId ? { ...group, repeat: Math.max(1, Math.round(repeat)) } : group)) };
}
