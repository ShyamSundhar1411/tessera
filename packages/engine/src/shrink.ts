import type { ArgValue, Model, Node } from "./types";

// Shrinking a language model to a size that trains on a free GPU in minutes,
// keeping its wiring: the same kind of attention, the same ratio of query to
// key/value heads, the same gated feed-forward or experts, the same norms.
// This is how architecture ideas are tested before anyone spends big compute.

export interface ShrinkOptions {
  /** Hidden size of the small version. */
  dim?: number;
  /** Total layers, shared out across the model's kinds of block. */
  layers?: number;
  /** Query heads. Key/value heads keep the original ratio. */
  heads?: number;
  /** Vocabulary of the tokenizer the small version will train with. */
  vocab?: number;
  /** Sequence length for training. */
  seqLen?: number;
  /** The most experts a mixture-of-experts layer keeps. */
  maxExperts?: number;
}

const round8 = (n: number) => Math.max(8, Math.round(n / 8) * 8);
const even = (n: number) => Math.max(2, Math.round(n / 2) * 2);

export function shrinkLanguageModel(model: Model, options: ShrinkOptions = {}): Model {
  if (model.kind !== "language") throw new Error("Only language models are shrunk; image models already train at small sizes.");
  const embedding = model.nodes.find((node) => node.op === "embedding");
  const originalDim = Number(embedding?.args.dim ?? 512);
  const dim = options.dim ?? 256;
  const ratio = dim / originalDim;
  const targetHeads = options.heads ?? 4;
  const seqLen = options.seqLen ?? 256;
  const maxExperts = options.maxExperts ?? 8;

  const nodes: Node[] = model.nodes.map((node) => {
    const a: Record<string, ArgValue> = { ...node.args };
    switch (node.op) {
      case "input":
        a.shape = String(seqLen);
        break;
      case "embedding":
        a.dim = dim;
        if (options.vocab) a.vocab = options.vocab;
        break;
      case "posembed":
        a.max = Math.max(seqLen, Math.min(Number(a.max ?? seqLen), seqLen * 4));
        break;
      case "lmhead":
        if (options.vocab) a.vocab = options.vocab;
        break;
      case "attention": {
        const heads = Number(a.heads);
        const kv = Number(a.kv ?? heads);
        const share = heads / kv;
        const h = Math.max(1, Math.min(targetHeads, heads));
        a.heads = h;
        // Keep the same sharing: Llama's 4 query heads per key/value head stays 4.
        a.kv = Math.max(1, Math.round(h / Math.min(share, h)));
        while (h % Number(a.kv)) a.kv = Number(a.kv) - 1;
        a.head_dim = even(dim / h);
        if (a.window) a.window = Math.max(16, Math.round(Number(a.window) * (seqLen / 4096)));
        break;
      }
      case "mla": {
        const h = Math.max(1, Math.min(targetHeads, Number(a.heads)));
        a.heads = h;
        if (Number(a.q_rank)) a.q_rank = round8(Number(a.q_rank) * ratio);
        a.kv_rank = round8(Number(a.kv_rank) * ratio);
        a.qk_nope_dim = even(Math.max(8, (Number(a.qk_nope_dim) * dim) / originalDim));
        a.qk_rope_dim = even(Math.max(8, (Number(a.qk_rope_dim) * dim) / originalDim));
        a.v_dim = even(Math.max(8, (Number(a.v_dim) * dim) / originalDim));
        break;
      }
      case "mlp":
        a.ff = round8(Number(a.ff) * ratio);
        break;
      case "moe": {
        const experts = Math.min(Number(a.experts), maxExperts);
        a.topk = Math.max(1, Math.min(Number(a.topk), Math.floor(experts / 2)));
        a.experts = experts;
        a.ff = round8(Number(a.ff) * ratio);
        if (a.shared_ff) a.shared_ff = round8(Number(a.shared_ff) * ratio);
        break;
      }
    }
    return { ...node, args: a };
  });

  // Share the layer budget across the repeated blocks, at least one of each kind.
  const repeated = model.groups.filter((group) => (group.repeat ?? 1) >= 1 && !group.parent);
  const total = repeated.reduce((sum, group) => sum + (group.repeat ?? 1), 0) || 1;
  const budget = Math.max(repeated.length, options.layers ?? 4);
  let left = budget;
  const groups = model.groups.map((group) => {
    if (!repeated.includes(group)) return { ...group };
    const share = Math.max(1, Math.round(((group.repeat ?? 1) / total) * budget));
    const repeat = Math.min(share, left - (repeated.length - repeated.indexOf(group) - 1));
    left -= repeat;
    return { ...group, repeat: Math.max(1, repeat) };
  });

  return {
    ...model,
    name: `${model.name} (small)`,
    nodes,
    groups,
    seqLen,
    meta: { ...model.meta, shrunk: `hidden ${dim}, ${budget} layers, from hidden ${originalDim}` },
  };
}
