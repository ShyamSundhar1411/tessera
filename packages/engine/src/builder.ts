import type { ArgValue, Group, Model, ModelKind, Node } from "./types";

/**
 * Builds a model node by node. Each call returns the new node's id, so code
 * that builds a network reads like the network: `x = b.add("conv2d", x, {...})`.
 */
export class Builder {
  private nodes: Node[] = [];
  private groups: Group[] = [];
  private counts = new Map<string, number>();
  private current: string | undefined;

  constructor(
    private name: string,
    private kind: ModelKind,
    private source = "catalog",
  ) {}

  private id(base: string): string {
    const n = (this.counts.get(base) ?? 0) + 1;
    this.counts.set(base, n);
    return n === 1 ? base : `${base}_${n}`;
  }

  input(shape: string, label = "Input"): string {
    const id = this.id("input");
    this.nodes.push({ id, op: "input", args: { shape }, inputs: [], label, group: this.current });
    return id;
  }

  node(op: string, inputs: string | string[], args: Record<string, ArgValue> = {}, label?: string): string {
    const id = this.id(op);
    this.nodes.push({ id, op, args, inputs: Array.isArray(inputs) ? inputs : [inputs], label, group: this.current });
    return id;
  }

  /** Runs `build` with every node it adds placed in a new block. */
  group<T>(label: string, kind: string, build: () => T, repeat?: number): T {
    const id = this.id(`g_${kind}`);
    this.groups.push({ id, label, kind, parent: this.current, repeat });
    const outer = this.current;
    this.current = id;
    try {
      return build();
    } finally {
      this.current = outer;
    }
  }

  build(meta: Model["meta"] = {}, seqLen?: number): Model {
    return { name: this.name, source: this.source, kind: this.kind, nodes: this.nodes, groups: this.groups, meta, seqLen };
  }
}
