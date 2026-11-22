import { analyze } from "../analyze";
import type { Analysis, Model, Node } from "../types";

// Writes an architecture as a PyTorch module, built from its connections.
// Building blocks the standard library lacks (grouped-query attention, latent
// attention, gated feed-forward, mixture of experts) are written out as small
// classes at the top, only when the model uses them. A block repeated N times
// becomes its own class in an nn.ModuleList.

const HELPERS: Record<string, string> = {
  rope: `def apply_rope(x, theta=10000.0):
    """Rotary position embedding on the last dimension of [batch, heads, seq, dim]."""
    seq, dim = x.shape[-2], x.shape[-1]
    half = dim // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    angles = torch.arange(seq, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)`,
  attention: `class Attention(nn.Module):
    """Self-attention, causal unless told otherwise. kv_heads < heads shares each key/value head across a group of query heads."""

    def __init__(self, dim, heads, kv_heads, head_dim, qkv_bias=False, o_bias=False, qk_norm=False, window=0, rope=True, causal=True):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.window, self.rope, self.causal = heads, kv_heads, head_dim, window, rope, causal
        self.q = nn.Linear(dim, heads * head_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.o = nn.Linear(heads * head_dim, dim, bias=o_bias)
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_norm(self.q(x).view(b, s, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(b, s, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        mask = None
        if self.window:
            i = torch.arange(s, device=x.device)
            mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal and mask is None, enable_gqa=self.kv_heads != self.heads)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))`,
  cross: `class CrossAttention(nn.Module):
    """Cross-attention: queries from the decoder stream, keys and values from the encoder output."""

    def __init__(self, dim, memory_dim, heads, kv_heads, head_dim, bias=False):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q = nn.Linear(dim, heads * head_dim, bias=bias)
        self.k = nn.Linear(memory_dim, kv_heads * head_dim, bias=bias)
        self.v = nn.Linear(memory_dim, kv_heads * head_dim, bias=bias)
        self.o = nn.Linear(heads * head_dim, dim, bias=bias)

    def forward(self, x, memory):
        b, s, _ = x.shape
        m = memory.shape[1]
        q = self.q(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.kv_heads != self.heads)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))`,
  sine: `class SinePositions(nn.Module):
    """The original Transformer's fixed positions: sine and cosine waves, no weights."""

    def forward(self, x):
        s, d = x.shape[1], x.shape[2]
        position = torch.arange(s, device=x.device, dtype=torch.float32)[:, None]
        rate = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / d))
        table = torch.zeros(s, d, device=x.device, dtype=x.dtype)
        table[:, 0::2], table[:, 1::2] = torch.sin(position * rate), torch.cos(position * rate)
        return x + table`,
  cls: `class ClassToken(nn.Module):
    """One learned token placed in front of the sequence (BERT, ViT)."""

    def __init__(self, dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return torch.cat([self.token.expand(x.shape[0], -1, -1), x], dim=1)`,
  mla: `class LatentAttention(nn.Module):
    """Multi-head latent attention (DeepSeek): keys and values pass through a small shared latent, which is all the cache keeps."""

    def __init__(self, dim, heads, q_rank, kv_rank, nope_dim, rope_dim, v_dim):
        super().__init__()
        self.heads, self.nope_dim, self.rope_dim, self.v_dim, self.kv_rank = heads, nope_dim, rope_dim, v_dim, kv_rank
        q_out = heads * (nope_dim + rope_dim)
        if q_rank:
            self.q = nn.Sequential(nn.Linear(dim, q_rank, bias=False), nn.RMSNorm(q_rank), nn.Linear(q_rank, q_out, bias=False))
        else:
            self.q = nn.Linear(dim, q_out, bias=False)
        self.kv_down = nn.Linear(dim, kv_rank + rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_rank)
        self.kv_up = nn.Linear(kv_rank, heads * (nope_dim + v_dim), bias=False)
        self.o = nn.Linear(heads * v_dim, dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.heads, self.nope_dim + self.rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        latent, k_rope = self.kv_down(x).split([self.kv_rank, self.rope_dim], dim=-1)
        kv = self.kv_up(self.kv_norm(latent)).view(b, s, self.heads, self.nope_dim + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        k_rope = apply_rope(k_rope.unsqueeze(1)).expand(-1, self.heads, -1, -1)
        q = torch.cat([q_nope, apply_rope(q_rope)], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))`,
  mlp: `class FeedForward(nn.Module):
    """A transformer's feed-forward network. Gated variants (SwiGLU, GeGLU) use three matrices."""

    def __init__(self, dim, hidden, act="swiglu", bias=False):
        super().__init__()
        self.gated = act in ("swiglu", "geglu")
        self.act = F.silu if act == "swiglu" else F.gelu if act in ("geglu", "gelu") else F.relu
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.gate = nn.Linear(dim, hidden, bias=False) if self.gated else None
        self.down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        h = self.act(self.gate(x)) * self.up(x) if self.gated else self.act(self.up(x))
        return self.down(h)`,
  moe: `class MixtureOfExperts(nn.Module):
    """A router picks top_k of many feed-forward experts for each token; shared experts see every token."""

    def __init__(self, dim, hidden, experts, top_k, shared=0, shared_hidden=0, shared_gate=False, router_bias=False):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(dim, experts, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(experts)) if router_bias else None
        self.experts = nn.ModuleList([FeedForward(dim, hidden) for _ in range(experts)])
        self.shared = nn.ModuleList([FeedForward(dim, shared_hidden or hidden) for _ in range(shared)])
        self.shared_gate = nn.Linear(dim, 1, bias=False) if shared_gate else None

    def forward(self, x):
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        scores = self.router(flat)
        if self.router_bias is not None:
            scores = scores + self.router_bias
        weights, chosen = scores.softmax(dim=-1).topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token, slot = (chosen == e).nonzero(as_tuple=True)
            if token.numel():
                out.index_add_(0, token, expert(flat[token]) * weights[token, slot, None])
        for expert in self.shared:
            extra = expert(flat)
            out = out + (torch.sigmoid(self.shared_gate(flat)) * extra if self.shared_gate is not None else extra)
        return out.view(b, s, d)`,
  types: `class TokenTypeEmbedding(nn.Module):
    """BERT's segment table: every token here is segment 0, so its first row is added to all of them."""

    def __init__(self, types, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(types, dim))

    def forward(self, x, segments=None):
        return x + (self.weight[segments] if segments is not None else self.weight[0])`,
  posembed: `class PositionEmbedding(nn.Module):
    """Learned position vectors added to the token vectors (GPT-2)."""

    def __init__(self, max_positions, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(max_positions, dim))

    def forward(self, x):
        return x + self.weight[: x.shape[1]]`,
};

const py = (id: string) => {
  const name = id.replace(/[^A-Za-z0-9_]/g, "_").replace(/^(\d)/, "n$1");
  return ["in", "def", "class", "return", "lambda", "input", "pass", "for", "if", "is", "not", "and", "or"].includes(name) ? `${name}_` : name;
};
const bool = (value: unknown) => (value === true || value === "true" ? "True" : "False");
const num = (value: unknown, fallback: number) => (value === undefined || value === "" ? fallback : Number(value));
const className = (label: string) =>
  label
    .replace(/[^A-Za-z0-9]+/g, " ")
    .trim()
    .split(" ")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join("")
    .replace(/^(\d)/, "Block$1") || "Block";

interface Ctx {
  analysis: Analysis;
  model: Model;
  needs: Set<string>;
  /** How forward() reaches the top-level module ("self"; inside a block class, the model is not reachable). */
  root: string;
}

/** The constructor for a node's module, or null when it is a plain function in forward(). */
function moduleFor(node: Node, ctx: Ctx): string | null {
  const shapeOf = (id: string) => ctx.analysis.info.get(id)?.shape ?? [];
  const input = node.inputs[0] ? shapeOf(node.inputs[0]) : [];
  const a = node.args;
  const features = input[input.length - 1];
  switch (node.op) {
    case "conv2d": {
      const inC = input[0];
      const groups = a.groups === "depthwise" ? inC : num(a.groups, 1);
      const k = num(a.k, 3);
      return `nn.Conv2d(${inC}, ${a.out}, kernel_size=${k}, stride=${num(a.s, 1)}, padding=${num(a.p, Math.floor(k / 2))}${groups !== 1 ? `, groups=${groups}` : ""}, bias=${bool(a.bias ?? true)})`;
    }
    case "batchnorm":
      return input.length === 3 ? `nn.BatchNorm2d(${input[0]})` : `nn.BatchNorm1d(${input[0]})`;
    case "layernorm":
      return `nn.LayerNorm(${features}${a.bias === false ? ", bias=False" : ""})`;
    case "rmsnorm":
      return `nn.RMSNorm(${features})`;
    case "dropout":
      return `nn.Dropout(${num(a.rate, 0.1)})`;
    case "maxpool":
      return `nn.MaxPool2d(kernel_size=${num(a.k, 2)}, stride=${num(a.s, num(a.k, 2))}, padding=${num(a.p, 0)}${a.ceil === true || a.ceil === "true" ? ", ceil_mode=True" : ""})`;
    case "avgpool":
      return `nn.AvgPool2d(kernel_size=${num(a.k, 2)}, stride=${num(a.s, num(a.k, 2))}, padding=${num(a.p, 0)}${a.ceil === true || a.ceil === "true" ? ", ceil_mode=True" : ""})`;
    case "upconv2d":
      return `nn.ConvTranspose2d(${input[0]}, ${a.out}, kernel_size=${num(a.k, 2)}, stride=${num(a.s, 2)}, padding=${num(a.p, 0)}, bias=${bool(a.bias ?? true)})`;
    case "linear":
      return `nn.Linear(${features}, ${a.out}, bias=${bool(a.bias ?? true)})`;
    case "embedding":
      // A tied embedding reuses another one's table, so it has no module of its own.
      return a.tied === true || a.tied === "true" ? null : `nn.Embedding(${a.vocab}, ${a.dim})`;
    case "posembed":
      if (a.fixed === true || a.fixed === "true") {
        ctx.needs.add("sine");
        return "SinePositions()";
      }
      if (a.types === true || a.types === "true") {
        ctx.needs.add("types");
        return `TokenTypeEmbedding(${num(a.max, 2)}, ${features})`;
      }
      ctx.needs.add("posembed");
      return `PositionEmbedding(${num(a.max, input[0])}, ${features})`;
    case "cross_attention": {
      ctx.needs.add("cross");
      const memory = shapeOf(node.inputs[1] ?? "");
      const heads = num(a.heads, 1);
      return `CrossAttention(${features}, ${memory[memory.length - 1]}, heads=${heads}, kv_heads=${num(a.kv, heads)}, head_dim=${num(a.head_dim, Math.floor(features / heads))}, bias=${bool(a.bias)})`;
    }
    case "cls_token":
      ctx.needs.add("cls");
      return `ClassToken(${features})`;
    case "attention": {
      ctx.needs.add("attention");
      ctx.needs.add("rope");
      const heads = num(a.heads, 1);
      return `Attention(${features}, heads=${heads}, kv_heads=${num(a.kv, heads)}, head_dim=${num(a.head_dim, Math.floor(features / heads))}, qkv_bias=${bool(a.qkv_bias)}, o_bias=${bool(a.o_bias)}, qk_norm=${bool(a.qk_norm)}, window=${num(a.window, 0)}, rope=${bool(a.rope ?? ctx.model.kind === "language")}, causal=${bool(a.causal ?? true)})`;
    }
    case "mla":
      ctx.needs.add("mla");
      ctx.needs.add("rope");
      return `LatentAttention(${features}, heads=${a.heads}, q_rank=${num(a.q_rank, 0)}, kv_rank=${a.kv_rank}, nope_dim=${a.qk_nope_dim}, rope_dim=${a.qk_rope_dim}, v_dim=${a.v_dim})`;
    case "mlp":
      ctx.needs.add("mlp");
      return `FeedForward(${features}, ${a.ff}, act="${a.act ?? "swiglu"}", bias=${bool(a.bias)})`;
    case "moe":
      ctx.needs.add("mlp");
      ctx.needs.add("moe");
      return `MixtureOfExperts(${features}, ${a.ff}, experts=${a.experts}, top_k=${num(a.topk, 2)}, shared=${num(a.shared, 0)}, shared_hidden=${num(a.shared_ff, 0)}, shared_gate=${bool(a.shared_gate)}, router_bias=${bool(a.router_bias)})`;
    case "lmhead":
      return `nn.Linear(${features}, ${a.vocab}, bias=False)`;
    default:
      return null;
  }
}

/** The forward() expression for a node, given its inputs' variable names. */
function expression(node: Node, inputs: string[], self: string, ctx: Ctx): string {
  const [x] = inputs;
  const rank = ctx.analysis.info.get(node.inputs[0] ?? "")?.shape.length ?? 1;
  switch (node.op) {
    case "relu":
      return `F.relu(${x})`;
    case "relu6":
      return `F.relu6(${x})`;
    case "gelu":
      return `F.gelu(${x})`;
    case "silu":
      return `F.silu(${x})`;
    case "sigmoid":
      return `torch.sigmoid(${x})`;
    case "tanh":
      return `torch.tanh(${x})`;
    case "hardswish":
      return `F.hardswish(${x})`;
    case "softmax":
      return `${x}.softmax(dim=-1)`;
    case "gap":
      return `F.adaptive_avg_pool2d(${x}, 1)`;
    case "flatten":
      return `torch.flatten(${x}, 1)`;
    case "identity":
      return x;
    case "to_sequence":
      return `${x}.flatten(2).transpose(1, 2)`;
    case "take_first":
      return `${x}[:, 0]`;
    case "mean_pool":
      return `${x}.mean(dim=1)`;
    case "cross_attention":
      return `${self}.${py(node.id)}(${inputs[0]}, ${inputs[1]})`;
    case "embedding":
      if (node.args.tied === true || node.args.tied === "true") {
        const table = ctx.model.nodes.find((item) => item.op === "embedding" && item.args.tied !== true && item.args.tied !== "true");
        return `${ctx.root}.${py(table?.id ?? node.id)}(${x})`;
      }
      return `${self}.${py(node.id)}(${x})`;
    case "add":
      return inputs.join(" + ");
    case "mul":
      return inputs.join(" * ");
    case "concat": {
      let axis = num(node.args.axis, rank === 3 ? 0 : rank - 1);
      if (axis < 0) axis += rank;
      return `torch.cat([${inputs.join(", ")}], dim=${axis + 1})`;
    }
    default:
      return `${self}.${py(node.id)}(${x})`;
  }
}

/** Lines of a forward pass over `nodes`, reading `entry` as the variable for the outside input. */
function forwardLines(nodes: Node[], ctx: Ctx, self: string, rename: Map<string, string>): string[] {
  const lines: string[] = [];
  for (const node of nodes) {
    if (node.op === "input") continue;
    const inputs = node.inputs.map((id) => rename.get(id) ?? py(id));
    lines.push(`${py(node.id)} = ${expression(node, inputs, self, ctx)}`);
  }
  return lines;
}

export interface ExportOptions {
  className?: string;
}

export function toPyTorch(model: Model, options: ExportOptions = {}): string {
  const analysis = analyze(model);
  const ctx: Ctx = { analysis, model, needs: new Set(), root: "self" };
  const name = options.className ?? className(model.name.split("/").pop() ?? "Model");
  const classes: string[] = [];
  const init: string[] = [];
  const forward: string[] = [];
  const inputs = model.nodes.filter((node) => node.op === "input");
  const input = inputs[0];
  const inputVar = model.kind === "language" ? "tokens" : "x";
  const rename = new Map<string, string>();
  // One input is x or tokens; several are named after their labels, such as source_tokens and target_tokens.
  const inputNames = inputs.map((node, i) => (inputs.length === 1 ? inputVar : py((node.label ?? `input_${i + 1}`).toLowerCase())));
  inputs.forEach((node, i) => rename.set(node.id, inputNames[i]));

  // Repeated top-level blocks become classes; everything else is inlined.
  const repeated = new Map(model.groups.filter((group) => (group.repeat ?? 1) > 1 && !group.parent).map((group) => [group.id, group]));
  const groupOf = (node: Node): string | undefined => {
    let id = node.group;
    while (id) {
      if (repeated.has(id)) return id;
      id = model.groups.find((group) => group.id === id)?.parent;
    }
    return undefined;
  };
  let lastLabel: string | undefined;
  const done = new Set<string>();

  for (const node of model.nodes) {
    if (node.op === "input") continue;
    const blockId = groupOf(node);
    if (blockId) {
      if (done.has(blockId)) continue;
      done.add(blockId);
      const group = repeated.get(blockId)!;
      const members = model.nodes.filter((item) => groupOf(item) === blockId);
      const inner = new Set(members.map((item) => item.id));
      // The block's main input, then anything else it reads from outside (a decoder reads the encoder output).
      const outside = [...new Set(members.flatMap((item) => item.inputs).filter((id) => !inner.has(id)))];
      const entry = outside[0];
      const extras = outside.slice(1);
      const blockClass = className(group.label);
      const blockInit = members.map((item) => [item, moduleFor(item, ctx)] as const).filter(([, mod]) => mod).map(([item, mod]) => `        self.${py(item.id)} = ${mod}`);
      const blockRename = new Map([[entry, "x"], ...extras.map((id, i) => [id, i === 0 ? "memory" : `extra_${i}`] as [string, string])]);
      const body = forwardLines(members, ctx, "self", blockRename).map((line) => `        ${line}`);
      const exit = py(members[members.length - 1].id);
      classes.push(`class ${blockClass}(nn.Module):
    """${group.label}, repeated ${group.repeat} times."""

    def __init__(self):
        super().__init__()
${blockInit.join("\n") || "        pass"}

    def forward(self, x${extras.map((_, i) => (i === 0 ? ", memory" : `, extra_${i}`)).join("")}):
${body.join("\n")}
        return ${exit}`);
      const list = py(`${group.label}_layers`).toLowerCase();
      init.push(`        self.${list} = nn.ModuleList([${blockClass}() for _ in range(${group.repeat})])`);
      const source = rename.get(entry) ?? py(entry);
      const extraArgs = extras.map((id) => `, ${rename.get(id) ?? py(id)}`).join("");
      forward.push(`        ${exit} = ${source}`, `        for layer in self.${list}:`, `            ${exit} = layer(${exit}${extraArgs})`);
      rename.set(members[members.length - 1].id, exit);
      continue;
    }
    const label = node.group ? model.groups.find((group) => group.id === node.group)?.label : undefined;
    if (label && label !== lastLabel) forward.push(`        # ${label}`);
    lastLabel = label;
    const mod = moduleFor(node, ctx);
    if (mod) init.push(`        self.${py(node.id)} = ${mod}`);
    forward.push(...forwardLines([node], ctx, "self", rename).map((line) => `        ${line}`));
  }

  // A tied output layer reuses the embedding's weights.
  const head = model.nodes.find((node) => node.op === "lmhead" && (node.args.tied === true || node.args.tied === "true"));
  const embedding = model.nodes.find((node) => node.op === "embedding");
  if (head && embedding) init.push(`        self.${py(head.id)}.weight = self.${py(embedding.id)}.weight  # tied to the embedding`);

  const last = model.nodes[model.nodes.length - 1];
  const result = last ? (rename.get(last.id) ?? py(last.id)) : inputVar;
  const helperOrder = ["rope", "sine", "types", "posembed", "cls", "attention", "cross", "mla", "mlp", "moe"].filter((key) => ctx.needs.has(key));
  const shape = input ? String(input.args.shape).split(/[x,]/).map(Number) : [];
  const large = analysis.params > 2e9;
  const example =
    model.kind === "language"
      ? inputs.map(() => `torch.randint(0, ${embedding?.args.vocab ?? 1000}, (1, 16))`).join(", ")
      : `torch.randn(1, ${shape.join(", ")})`;

  return `"""
${model.name}
Written by Tessera from the architecture's connections.
Parameters: ${analysis.params.toLocaleString("en-US")}${analysis.activeParams !== analysis.params ? ` (${analysis.activeParams.toLocaleString("en-US")} active per token)` : ""}
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

${[...helperOrder.map((key) => HELPERS[key]), ...classes].join("\n\n\n")}${helperOrder.length || classes.length ? "\n\n\n" : ""}class ${name}(nn.Module):
    def __init__(self):
        super().__init__()
${init.join("\n") || "        pass"}

    def forward(self, ${inputNames.join(", ") || inputVar}):
${forward.join("\n")}
        return ${result}


if __name__ == "__main__":
${
  large
    ? `    # ${(analysis.params / 1e9).toFixed(1)}B parameters do not fit in most memory, so build it on the
    # "meta" device: shapes and counts without allocating the weights.
    with torch.device("meta"):
        model = ${name}()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")`
    : `    model = ${name}()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    out = model(${example})
    print("output shape:", tuple(out.shape))`
}
`;
}
