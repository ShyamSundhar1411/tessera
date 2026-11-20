import { ALIASES, OPS } from "./ops";
import type { ArgValue, Group, Model, ModelKind, Node } from "./types";

// The text format: an architecture written a line at a time.
//
//   model "Small CNN with a skip"
//   input 3x32x32
//   stem = conv 32 k3 -> bn -> relu
//   y = conv 32 -> bn -> relu -> conv 32 -> bn
//   join = add(y, stem) -> relu
//   block "Decoder layer" x32 {
//     h = norm -> attn 32 kv=8
//     mid = add(h, @in)
//   }
//   head = pool -> linear 10
//
// A chain continues from the line above unless it starts with @name or a
// join such as add(a, b). `name =` names the chain's last node so later
// lines can use it. Inside a block, @in is the block's input. A block with
// xN is repeated N times and must keep its shape.

export interface ParseError {
  line: number;
  message: string;
}

export interface ParseResult {
  model: Model;
  errors: ParseError[];
}

const SHORT_KEYS: Record<string, string> = { k: "k", s: "s", p: "p", g: "groups" };
const JOINS = new Set(["add", "concat", "mul", "cat"]);

const value = (raw: string): ArgValue => {
  if (raw === "true") return true;
  if (raw === "false") return false;
  const n = Number(raw);
  return raw !== "" && Number.isFinite(n) ? n : raw.replace(/^"|"$/g, "");
};

/** Splits a line into tokens, keeping quoted strings and bracketed calls together. */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let depth = 0;
  let quoted = false;
  for (const ch of text) {
    if (ch === '"') quoted = !quoted;
    if (!quoted && ch === "(") depth++;
    if (!quoted && ch === ")") depth--;
    if (!quoted && depth === 0 && /\s/.test(ch)) {
      if (current) tokens.push(current);
      current = "";
    } else current += ch;
  }
  if (current) tokens.push(current);
  return tokens;
}

export function parseText(source: string): ParseResult {
  const nodes: Node[] = [];
  const groups: Group[] = [];
  const errors: ParseError[] = [];
  const names = new Map<string, string>();
  const counts = new Map<string, number>();
  let name = "Untitled architecture";
  let kind: ModelKind | null = null;
  let seqLen: number | undefined;
  let previous: string | null = null;
  const stack: { group: Group; input: string | null }[] = [];

  const freshId = (base: string) => {
    let n = (counts.get(base) ?? 0) + 1;
    while (names.has(n === 1 ? base : `${base}_${n}`)) n++;
    counts.set(base, n);
    return n === 1 ? base : `${base}_${n}`;
  };
  const resolve = (ref: string, line: number): string | null => {
    const key = ref.replace(/^@/, "");
    if (key === "in") {
      const top = stack[stack.length - 1];
      if (!top) {
        errors.push({ line, message: "@in only works inside a block" });
        return null;
      }
      return top.input;
    }
    const id = names.get(key);
    if (!id) errors.push({ line, message: `nothing is called "${key}" yet` });
    return id ?? null;
  };

  source.split("\n").forEach((rawLine, index) => {
    const line = index + 1;
    const text = rawLine.replace(/#.*$/, "").trim();
    if (!text) return;

    let m: RegExpMatchArray | null;
    if ((m = text.match(/^model\s+"?([^"]+)"?$/))) {
      name = m[1].trim();
      return;
    }
    if ((m = text.match(/^kind\s+(vision|language|tabular)$/))) {
      kind = m[1] as ModelKind;
      return;
    }
    if ((m = text.match(/^input\s+(?:(?:image|tokens|vector)\s+)?([\dx×,\s]+)$/))) {
      const shape = m[1].replace(/\s+/g, "").replace(/×/g, "x");
      const id = freshId("input");
      names.set("input", id);
      nodes.push({ id, op: "input", args: { shape }, inputs: [], label: "Input", line });
      previous = id;
      const dims = shape.split(/[x,]/).filter(Boolean);
      if (!kind) kind = dims.length === 3 ? "vision" : dims.length === 1 && /tokens/.test(text) ? "language" : dims.length === 1 ? "tabular" : "tabular";
      if (dims.length === 1 && /tokens/.test(text)) seqLen = Number(dims[0]);
      return;
    }
    if ((m = text.match(/^block\s+"([^"]+)"(?:\s+x(\d+))?(?:\s+kind=(\S+))?\s*\{$/))) {
      const group: Group = { id: freshId(`g_${m[1].toLowerCase().replace(/[^a-z0-9]+/g, "_")}`), label: m[1], kind: m[3] ?? m[1].toLowerCase().replace(/[^a-z0-9]+/g, "-"), parent: stack[stack.length - 1]?.group.id };
      if (m[2]) group.repeat = Number(m[2]);
      groups.push(group);
      stack.push({ group, input: previous });
      return;
    }
    if (text === "}") {
      if (!stack.pop()) errors.push({ line, message: "a } with no block to close" });
      return;
    }

    // A chain, optionally named: name = start -> op -> op
    let target: string | null = null;
    let body = text;
    if ((m = text.match(/^([A-Za-z_][\w]*)\s*=\s*(.+)$/))) {
      target = m[1];
      body = m[2];
    }
    const steps = body.split("->").map((step) => step.trim()).filter(Boolean);
    let current: string | null = previous;
    let created: Node | null = null;

    steps.forEach((step, stepIndex) => {
      const tokens = tokenize(step);
      const head = tokens[0] ?? "";
      // A reference to an earlier node starts the chain there.
      if (head.startsWith("@") && tokens.length === 1) {
        if (stepIndex !== 0) errors.push({ line, message: "an @reference can only start a chain" });
        current = resolve(head, line);
        return;
      }
      // A join: add(a, b), concat(a, b, axis=1), mul(a, b)
      const call = step.match(/^(\w+)\((.*)\)\s*(.*)$/);
      if (call && JOINS.has(call[1])) {
        const op = ALIASES[call[1]] ?? call[1];
        const parts = call[2].split(",").map((part) => part.trim()).filter(Boolean);
        const args: Record<string, ArgValue> = {};
        const inputs: string[] = [];
        for (const part of parts) {
          const kv = part.match(/^(\w+)=(.+)$/);
          if (kv) args[kv[1]] = value(kv[2]);
          else {
            const id = resolve(part, line);
            if (id) inputs.push(id);
          }
        }
        const label = call[3].match(/^"(.+)"$/)?.[1];
        created = { id: freshId(op), op, args, inputs, line, group: stack[stack.length - 1]?.group.id, ...(label ? { label } : {}) };
        nodes.push(created);
        current = created.id;
        return;
      }
      const op = ALIASES[head] ?? head;
      const spec = OPS[op];
      if (!spec || op === "input") {
        errors.push({ line, message: `"${head}" is not an operation I know` });
        current = null;
        return;
      }
      const args: Record<string, ArgValue> = {};
      let label: string | undefined;
      for (const token of tokens.slice(1)) {
        let t: RegExpMatchArray | null;
        if ((t = token.match(/^"(.*)"$/))) label = t[1];
        else if ((t = token.match(/^(\w+)=(.+)$/))) args[t[1]] = value(t[2]);
        else if ((t = token.match(/^([kspg])(\d+)$/))) args[SHORT_KEYS[t[1]]] = Number(t[2]);
        else if (/^-?\d+(\.\d+)?$/.test(token) && spec.main) args[spec.main] = Number(token);
        else if (token === "nobias") args.bias = false;
        else if (token === "depthwise") args.groups = "depthwise";
        else if (/^[a-z_]+$/.test(token)) args[token] = true;
        else errors.push({ line, message: `I can't read "${token}" in ${head}` });
      }
      if (current === null && stepIndex === 0 && !previous) {
        errors.push({ line, message: "start with an input line, such as input 3x32x32" });
        return;
      }
      created = { id: freshId(op), op, args, inputs: current ? [current] : [], line, group: stack[stack.length - 1]?.group.id, ...(label ? { label } : {}) };
      nodes.push(created);
      current = created.id;
    });

    if (target && created) {
      const node = created as Node;
      if (names.has(target)) errors.push({ line, message: `"${target}" is already used` });
      // The name becomes the node's id when it is free, so exported code and
      // errors use it too. Nothing reads the node yet, so renaming is safe.
      if (!nodes.some((other) => other.id === target)) {
        if (current === node.id) current = target;
        node.id = target;
      }
      names.set(target, node.id);
    }
    if (current) previous = current;
  });

  if (stack.length) errors.push({ line: source.split("\n").length, message: `block "${stack[stack.length - 1].group.label}" is never closed with }` });
  return { model: { name, source: "text", kind: kind ?? "tabular", nodes, groups, meta: { family: "Custom" }, seqLen }, errors };
}

// -- Writing a model back as text ---------------------------------------------

const formatArg = (key: string, raw: ArgValue): string => {
  if (raw === true) return key;
  if (raw === "depthwise" && key === "groups") return "depthwise";
  if (key === "bias" && raw === false) return "nobias";
  if (["k", "s", "p"].includes(key) && typeof raw === "number") return `${key}${raw}`;
  if (key === "groups" && typeof raw === "number") return `g${raw}`;
  return `${key}=${typeof raw === "string" && /\s/.test(raw) ? `"${raw}"` : raw}`;
};

function formatStep(node: Node): string {
  const spec = OPS[node.op];
  const parts: string[] = [node.op];
  const main = spec?.main;
  if (main && node.args[main] !== undefined) parts.push(String(node.args[main]));
  for (const [key, raw] of Object.entries(node.args)) {
    if (key === main || raw === undefined || raw === false) {
      if (key === "bias" && raw === false) parts.push("nobias");
      continue;
    }
    parts.push(formatArg(key, raw));
  }
  if (node.label && node.label !== node.op) parts.push(`"${node.label.replace(/"/g, "'")}"`);
  return parts.join(" ");
}

/**
 * Writes a model as text that parses back to the same graph. Straight runs
 * of single-input nodes become one chain; a node read by more than its next
 * neighbour gets a name.
 */
export function toText(model: Model): string {
  const lines: string[] = [`model "${model.name}"`];
  const readers = new Map<string, number>();
  for (const node of model.nodes) for (const id of node.inputs) readers.set(id, (readers.get(id) ?? 0) + 1);
  const ident = (id: string) => id.replace(/[^A-Za-z0-9_]/g, "_").replace(/^(\d)/, "n$1");
  const groupOf = new Map(model.groups.map((group) => [group.id, group]));
  const chainOf = (groupId?: string): string[] => {
    const out: string[] = [];
    let g = groupId ? groupOf.get(groupId) : undefined;
    while (g) {
      out.unshift(g.id);
      g = g.parent ? groupOf.get(g.parent) : undefined;
    }
    return out;
  };

  let open: string[] = [];
  let previous: string | null = null;
  let chain: string[] = [];
  let chainName: string | null = null;
  const blockInputs = new Map<string, string | null>();
  const flush = () => {
    if (!chain.length) return;
    lines.push(`${"  ".repeat(open.length)}${chainName ? `${chainName} = ` : ""}${chain.join(" -> ")}`);
    chain = [];
    chainName = null;
  };

  for (const node of model.nodes) {
    const wanted = chainOf(node.group);
    // Close and open blocks to match this node's place.
    let common = 0;
    while (common < open.length && common < wanted.length && open[common] === wanted[common]) common++;
    if (common < open.length || wanted.length > common) flush();
    while (open.length > common) {
      open.pop();
      lines.push(`${"  ".repeat(open.length)}}`);
    }
    while (open.length < wanted.length) {
      const group = groupOf.get(wanted[open.length])!;
      lines.push(`${"  ".repeat(open.length)}block "${group.label}"${(group.repeat ?? 1) > 1 ? ` x${group.repeat}` : ""} kind=${group.kind} {`);
      blockInputs.set(group.id, previous);
      open.push(group.id);
    }

    if (node.op === "input") {
      flush();
      lines.push(`input ${model.kind === "language" ? "tokens " : ""}${String(node.args.shape)}`);
      previous = node.id;
      continue;
    }
    const top = open[open.length - 1];
    const ref = (id: string) => (top && blockInputs.get(top) === id ? "@in" : `@${ident(id)}`);
    const isJoin = OPS[node.op]?.arity === "many";
    const continues = !isJoin && node.inputs.length === 1 && node.inputs[0] === previous && chain.length > 0;
    let step = isJoin ? `${node.op}(${node.inputs.map((id) => ref(id).replace(/^@(?!in$)/, "")).join(", ")}${Object.entries(node.args).map(([k, v]) => `, ${k}=${v}`).join("")})${node.label ? ` "${node.label}"` : ""}` : formatStep(node);
    if (!continues) {
      flush();
      if (!isJoin && node.inputs.length === 1 && node.inputs[0] !== previous) step = `${ref(node.inputs[0])} -> ${step}`;
    }
    chain.push(step);
    // A node read by others later needs a name; naming ends the chain here.
    const used = readers.get(node.id) ?? 0;
    const next = model.nodes[model.nodes.indexOf(node) + 1];
    const readByNextOnly = used === 1 && next && next.inputs.length === 1 && next.inputs[0] === node.id && next.group === node.group;
    if (used > 0 && !readByNextOnly) {
      chainName = ident(node.id);
      flush();
    }
    previous = node.id;
  }
  flush();
  while (open.length) {
    open.pop();
    lines.push(`${"  ".repeat(open.length)}}`);
  }
  return lines.join("\n") + "\n";
}
