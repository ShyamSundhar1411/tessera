import ELK, { type ElkExtendedEdge, type ElkNode } from "elkjs/lib/elk.bundled.js";
import { OPS, timesOf, type Analysis, type Category, type ChangeKind, type Model, type Node, type Shape } from "tessera-engine";

// What the 2D and 3D views draw: a model with closed blocks folded into one
// box, open blocks drawn as containers around their contents, edges between
// whatever is visible, and every position worked out by ELK.

export const CATEGORY_COLOURS: Record<Category | "block", string> = {
  input: "#9aa2b4",
  conv: "#5fb4ff",
  linear: "#a78bfa",
  norm: "#7dd3c0",
  activation: "#f7b955",
  pool: "#60a5fa",
  merge: "#f0569f",
  reshape: "#94a3b8",
  embedding: "#34d399",
  attention: "#fb7185",
  mlp: "#c084fc",
  output: "#fbbf24",
  regularise: "#64748b",
  block: "#f6782a",
};

export interface DisplayNode {
  id: string;
  kind: "node" | "block";
  label: string;
  /** The operation, for single nodes. */
  op?: string;
  category: Category | "block";
  shape: Shape;
  params: number;
  /** For blocks: how many times it repeats, and whether it is open. */
  repeat: number;
  open: boolean;
  /** The open block this sits inside, if any. */
  parent?: string;
  /** How many nodes a closed block holds. */
  size: number;
  error?: string;
  status?: ChangeKind;
  /** Position and size from the layout, relative to the parent. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Absolute position, for the 3D view. */
  ax: number;
  ay: number;
  /** Order along the flow of data, for the 3D view's depth. */
  depth: number;
}

export interface DisplayEdge {
  id: string;
  source: string;
  target: string;
  /** A connection that jumps past its neighbours: a skip, shortcut or residual. */
  skip: boolean;
}

export interface Display {
  nodes: DisplayNode[];
  edges: DisplayEdge[];
  width: number;
  height: number;
}

const NODE = { width: 210, height: 58 };
const BLOCK = { width: 230, height: 66 };
const elk = new ELK();

export const formatShape = (shape: Shape, model: Model) => {
  if (!shape.length) return "";
  const parts = shape.map((n, i) => (model.kind === "language" && i === 0 && model.seqLen === n ? "seq" : String(n)));
  return parts.join(" × ");
};

/** Builds and lays out what to draw. `status` colours units by a diff, keyed by top-level block or node id. */
export async function buildDisplay(model: Model, analysis: Analysis, expanded: string[], status?: Map<string, ChangeKind>): Promise<Display> {
  const groups = new Map(model.groups.map((group) => [group.id, group]));
  const open = new Set(expanded);
  const parentOf = (groupId: string) => groups.get(groupId)?.parent;

  // The visible box for a node: itself if all its blocks are open, otherwise its outermost closed block.
  const visibleFor = (node: Node): string => {
    const chain: string[] = [];
    let id = node.group;
    while (id) {
      chain.unshift(id);
      id = parentOf(id);
    }
    for (const groupId of chain) if (!open.has(groupId)) return groupId;
    return node.id;
  };
  const topUnit = (node: Node): string => {
    let id = node.group;
    let top = id;
    while (id) {
      top = id;
      id = parentOf(id);
    }
    return top ?? node.id;
  };

  const shown = new Map<string, DisplayNode>();
  const order: string[] = [];
  model.nodes.forEach((node, index) => {
    const id = visibleFor(node);
    const info = analysis.info.get(node.id);
    const unitStatus = status?.get(topUnit(node));
    const existing = shown.get(id);
    if (existing) {
      existing.params += (info?.params ?? 0) * (info?.times ?? 1);
      existing.size += 1;
      if (info?.shape.length) existing.shape = info.shape;
      if (info?.error && !existing.error) existing.error = info.error;
      return;
    }
    const isBlock = id !== node.id;
    const group = isBlock ? groups.get(id)! : undefined;
    // An open block containing this box, if any.
    let parent = isBlock ? parentOf(id) : node.group;
    while (parent && !open.has(parent)) parent = parentOf(parent);
    const spec = OPS[node.op];
    shown.set(id, {
      id,
      kind: isBlock ? "block" : "node",
      label: isBlock ? group!.label : (node.label ?? node.op),
      op: isBlock ? undefined : node.op,
      category: isBlock ? "block" : (spec?.category ?? "reshape"),
      shape: info?.shape ?? [],
      params: (info?.params ?? 0) * (info?.times ?? 1),
      repeat: isBlock ? timesOf(model.groups, id) : 1,
      open: false,
      parent,
      size: 1,
      error: info?.error && info.error !== "an input above has a problem" ? info.error : undefined,
      status: unitStatus,
      x: 0,
      y: 0,
      width: isBlock ? BLOCK.width : NODE.width,
      height: isBlock ? BLOCK.height : NODE.height,
      ax: 0,
      ay: 0,
      depth: index,
    });
    order.push(id);
  });

  // Open blocks become containers.
  const containers = new Map<string, DisplayNode>();
  for (const groupId of open) {
    const group = groups.get(groupId);
    if (!group) continue;
    // Only open blocks whose ancestors are all open are actually shown.
    let ancestor = group.parent;
    let visible = true;
    while (ancestor) {
      if (!open.has(ancestor)) visible = false;
      ancestor = parentOf(ancestor);
    }
    if (!visible || ![...shown.values()].some((node) => node.parent === groupId)) continue;
    let parent = group.parent;
    while (parent && !open.has(parent)) parent = parentOf(parent);
    const members = model.nodes.filter((node) => {
      let id = node.group;
      while (id) {
        if (id === groupId) return true;
        id = parentOf(id);
      }
      return false;
    });
    containers.set(groupId, {
      id: groupId,
      kind: "block",
      label: group.label,
      category: "block",
      shape: [],
      params: members.reduce((sum, node) => sum + (analysis.info.get(node.id)?.params ?? 0) * (analysis.info.get(node.id)?.times ?? 1), 0),
      repeat: timesOf(model.groups, groupId),
      open: true,
      parent,
      size: members.length,
      status: status?.get(members[0] ? topUnit(members[0]) : groupId),
      x: 0,
      y: 0,
      width: 0,
      height: 0,
      ax: 0,
      ay: 0,
      depth: members[0] ? model.nodes.indexOf(members[0]) : 0,
    });
  }

  // Edges between visible boxes.
  const edges = new Map<string, DisplayEdge>();
  const position = new Map(order.map((id, i) => [id, i]));
  for (const node of model.nodes) {
    const target = visibleFor(node);
    for (const input of node.inputs) {
      const sourceNode = model.nodes.find((item) => item.id === input);
      if (!sourceNode) continue;
      const source = visibleFor(sourceNode);
      if (source === target) continue;
      const id = `${source}->${target}`;
      if (edges.has(id)) continue;
      const gap = (position.get(target) ?? 0) - (position.get(source) ?? 0);
      edges.set(id, { id, source, target, skip: gap > 1 && OPS[node.op]?.arity === "many" });
    }
  }

  // Lay out with ELK: open blocks hold their contents.
  const elkNode = (item: DisplayNode): ElkNode => {
    const children = [...shown.values(), ...containers.values()].filter((child) => child.parent === item.id);
    return item.open
      ? {
          id: item.id,
          layoutOptions: { "elk.padding": "[top=40,left=18,bottom=18,right=18]" },
          children: children.sort((a, b) => a.depth - b.depth).map(elkNode),
        }
      : { id: item.id, width: item.width, height: item.height };
  };
  const roots = [...shown.values(), ...containers.values()].filter((item) => !item.parent).sort((a, b) => a.depth - b.depth);
  const graph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "34",
      "elk.spacing.nodeNode": "28",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.edgeRouting": "SPLINES",
    },
    children: roots.map(elkNode),
    edges: [...edges.values()].map((edge): ElkExtendedEdge => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
  };
  const laid = await elk.layout(graph);

  const all = new Map([...shown, ...containers]);
  const place = (node: ElkNode, ox: number, oy: number) => {
    for (const child of node.children ?? []) {
      const item = all.get(child.id);
      if (!item) continue;
      item.x = child.x ?? 0;
      item.y = child.y ?? 0;
      item.width = child.width ?? item.width;
      item.height = child.height ?? item.height;
      item.ax = ox + item.x;
      item.ay = oy + item.y;
      place(child, item.ax, item.ay);
    }
  };
  place(laid, 0, 0);

  // Parents first, as React Flow needs.
  const depthOf = (item: DisplayNode): number => (item.parent ? 1 + depthOf(all.get(item.parent)!) : 0);
  const nodes = [...all.values()].sort((a, b) => depthOf(a) - depthOf(b) || a.depth - b.depth);
  return { nodes, edges: [...edges.values()], width: laid.width ?? 0, height: laid.height ?? 0 };
}
