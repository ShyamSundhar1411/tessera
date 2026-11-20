import { OPS, ShapeError } from "./ops";
import type { Analysis, Group, Model, NodeInfo } from "./types";

/** How many times a group runs: its repeat times its parents' repeats. */
export function timesOf(groups: Group[], id: string | undefined): number {
  let times = 1;
  const seen = new Set<string>();
  while (id && !seen.has(id)) {
    seen.add(id);
    const group = groups.find((item) => item.id === id);
    if (!group) break;
    times *= group.repeat ?? 1;
    id = group.parent;
  }
  return times;
}

/**
 * Works out every node's output shape, parameters and compute, and totals for
 * the model. A node that cannot be worked out records an error and stops its
 * descendants, so one mistake is reported once, where it happens.
 */
export function analyze(model: Model): Analysis {
  const info = new Map<string, NodeInfo>();
  const errors: Analysis["errors"] = [];
  const ids = new Set(model.nodes.map((node) => node.id));
  let params = 0;
  let activeParams = 0;
  let flops = 0;
  let last: string | null = null;

  for (const node of model.nodes) {
    const times = timesOf(model.groups, node.group);
    const spec = OPS[node.op];
    const fail = (message: string) => {
      info.set(node.id, { shape: [], params: 0, activeParams: 0, flops: 0, times, error: message });
      errors.push({ node: node.id, message, line: node.line });
    };
    if (!spec) {
      fail(`unknown operation "${node.op}"`);
      continue;
    }
    const missing = node.inputs.find((id) => !ids.has(id));
    if (missing) {
      fail(`reads from "${missing}", which does not exist`);
      continue;
    }
    if (spec.arity === "one" && node.inputs.length !== 1) {
      fail(`${node.op} takes one input, got ${node.inputs.length}`);
      continue;
    }
    if (spec.arity === "many" && node.inputs.length < 2) {
      fail(`${node.op} joins two or more inputs, got ${node.inputs.length}`);
      continue;
    }
    const upstream = node.inputs.map((id) => info.get(id));
    if (upstream.some((item) => !item)) {
      fail("reads from a node defined after it");
      continue;
    }
    if (upstream.some((item) => item!.error)) {
      // Already reported upstream; mark this one without repeating the message.
      info.set(node.id, { shape: [], params: 0, activeParams: 0, flops: 0, times, error: "an input above has a problem" });
      continue;
    }
    try {
      const result = spec.infer(
        upstream.map((item) => item!.shape),
        node.args,
      );
      const active = result.activeParams ?? result.params;
      info.set(node.id, { shape: result.shape, params: result.params, activeParams: active, flops: result.flops, times });
      params += result.params * times;
      activeParams += active * times;
      flops += result.flops * times;
      last = node.id;
    } catch (error) {
      fail(error instanceof ShapeError || error instanceof Error ? error.message : String(error));
    }
  }

  // A repeated block must hand on the same shape it receives.
  for (const group of model.groups) {
    if ((group.repeat ?? 1) <= 1) continue;
    const members = model.nodes.filter((node) => node.group === group.id);
    const inner = new Set(members.map((node) => node.id));
    const entry = members.flatMap((node) => node.inputs).find((id) => !inner.has(id));
    const exit = members[members.length - 1];
    const a = entry ? info.get(entry) : undefined;
    const b = exit ? info.get(exit.id) : undefined;
    if (a && b && !a.error && !b.error && a.shape.join() !== b.shape.join())
      errors.push({ node: exit.id, message: `block "${group.label}" repeats ${group.repeat} times, so it must keep its shape, but turns [${a.shape.join(", ")}] into [${b.shape.join(", ")}]`, line: exit.line });
  }

  return { info, params, activeParams, flops, outputShape: last ? info.get(last)!.shape : null, errors };
}
