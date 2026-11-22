import { Builder } from "../builder";
import type { Model } from "../types";

// Turns a Hugging Face config.json into the engine's graph. A config holds
// only settings; the wiring lives in each family's modelling code, so the
// families are encoded here. Most open models are the same decoder with a few
// switches (bias, norms, gating, experts, latent attention), so one template
// with those switches covers them.

type Config = Record<string, unknown>;

const n = (config: Config, ...keys: string[]): number | undefined => {
  for (const key of keys) {
    const value = config[key];
    if (typeof value === "number") return value;
  }
  return undefined;
};

export const SUPPORTED_FAMILIES = [
  "llama",
  "mistral",
  "mixtral",
  "qwen2",
  "qwen2_moe",
  "qwen3",
  "qwen3_moe",
  "phi3",
  "gemma",
  "gemma2",
  "gemma3_text",
  "deepseek_v2",
  "deepseek_v3",
  "gpt2",
  "bert",
  "roberta",
  "xlm-roberta",
  "distilbert",
  "olmo2",
  "granite",
  "smollm3",
];

export class ConfigError extends Error {}

/** Reads the address of a config file for a model on the Hugging Face hub. */
export const configUrl = (repo: string, revision = "main") => `https://huggingface.co/${repo.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/$/, "")}/resolve/${revision}/config.json`;

/** Fetches a public model's config. Gated models (Llama, Gemma) need it pasted instead. */
export async function fetchConfig(repo: string): Promise<Config> {
  const response = await fetch(configUrl(repo));
  if (response.status === 401 || response.status === 403)
    throw new ConfigError(`${repo} is gated on Hugging Face. Open its page, accept the licence, then paste its config.json here, or use a public copy.`);
  if (!response.ok) throw new ConfigError(`No config.json found for ${repo} (${response.status}).`);
  return (await response.json()) as Config;
}

const ENCODERS = new Set(["bert", "roberta", "xlm-roberta", "distilbert", "electra"]);

/**
 * BERT and its family: bidirectional encoders. Learned positions and token
 * types, normalisation after each part, GELU feed-forward with biases, and a
 * pooler on the first token. Counts match the base models in transformers
 * (BertModel, RobertaModel, DistilBertModel).
 */
function encoderFromConfig(config: Config, type: string, repo: string, seqLen: number): Model {
  const distil = type === "distilbert";
  const d = n(config, "hidden_size", "dim")!;
  const layers = n(config, "num_hidden_layers", "n_layers")!;
  const heads = n(config, "num_attention_heads", "n_heads")!;
  const ff = n(config, "intermediate_size", "hidden_dim") ?? 4 * d;
  const vocab = n(config, "vocab_size")!;
  const positions = n(config, "max_position_embeddings") ?? 512;
  const types = distil ? 0 : (n(config, "type_vocab_size") ?? 0);
  const b = new Builder(repo || type, "language", repo ? `huggingface:${repo}` : "huggingface");
  let x = b.input(String(Math.min(seqLen, positions)), "Tokens");
  x = b.node("embedding", x, { vocab, dim: d }, "Token embedding");
  x = b.node("posembed", x, { max: positions }, "Position embedding");
  if (types) x = b.node("posembed", x, { max: types, types: true }, "Token type embedding");
  x = b.node("layernorm", x, {}, "Embedding norm");
  const input = x;
  x = b.group(
    "Encoder layer",
    "encoder",
    () => {
      let h = b.node("attention", input, { heads, causal: false, qkv_bias: true, o_bias: true, rope: false }, "Self-attention (bidirectional)");
      h = b.node("add", [h, input], {}, "Residual");
      const mid = b.node("layernorm", h, {}, "Norm");
      let f = b.node("mlp", mid, { ff, act: "gelu", bias: true }, "Feed-forward");
      f = b.node("add", [f, mid], {}, "Residual");
      return b.node("layernorm", f, {}, "Norm");
    },
    layers,
  );
  if (!distil) {
    x = b.node("take_first", x, {}, "First token");
    x = b.node("linear", x, { out: d }, "Pooler");
    b.node("tanh", x);
  }
  return b.build(
    {
      family: type,
      layers,
      "hidden size": d,
      attention: `Bidirectional, ${heads} heads`,
      "feed-forward": `GELU, ${ff} wide`,
      vocab,
      context: positions,
      positions: "learned",
      norm: "LayerNorm, after each part",
      pooler: distil ? "none" : "first token, dense + tanh",
    },
    Math.min(seqLen, positions),
  );
}

export function fromHuggingFace(raw: Config, repo = "", seqLen = 2048): Model {
  // Multimodal configs keep the language model under text_config.
  const config: Config = typeof raw.text_config === "object" && raw.text_config ? { ...(raw.text_config as Config), model_type: (raw.text_config as Config).model_type ?? raw.model_type } : raw;
  const type = String(config.model_type ?? "");
  if (!type) throw new ConfigError("This config has no model_type, so its family is unknown.");
  if (ENCODERS.has(type)) return encoderFromConfig(config, type, repo, seqLen);

  const gpt2 = type === "gpt2";
  const d = n(config, "hidden_size", "n_embd", "d_model");
  const layers = n(config, "num_hidden_layers", "n_layer", "num_layers");
  const heads = n(config, "num_attention_heads", "n_head");
  if (!d || !layers || !heads) throw new ConfigError(`Missing hidden size, layer count or head count for ${type}.`);
  const kv = n(config, "num_key_value_heads") ?? heads;
  const headDim = n(config, "head_dim") ?? Math.floor(d / heads);
  const vocab = n(config, "vocab_size")!;
  const ff = n(config, "intermediate_size", "n_inner") ?? 4 * d;
  const context = n(config, "max_position_embeddings", "n_positions") ?? seqLen;
  const gemma = type.startsWith("gemma");
  const deepseek = type.startsWith("deepseek");
  // Transformers ties the output to the embedding unless a config says otherwise.
  const tied = typeof config.tie_word_embeddings === "boolean" ? config.tie_word_embeddings : true;
  const act = String(config.hidden_activation ?? config.hidden_act ?? config.activation_function ?? "silu");
  const gated = !gpt2;
  const mlpAct = !gated ? "gelu" : act.startsWith("gelu") ? "geglu" : "swiglu";
  const qkvBias = type === "qwen2" || type === "qwen2_moe" || gpt2 || config.attention_bias === true;
  const oBias = gpt2 || config.attention_bias === true;
  const qkNorm = type === "qwen3" || type === "qwen3_moe" || type === "gemma3_text" || type === "olmo2";
  const sandwich = type === "gemma2" || type === "gemma3_text";
  const useWindow = config.use_sliding_window !== false && typeof config.sliding_window === "number" && (type === "mistral" || type === "phi3");
  const window = useWindow ? (config.sliding_window as number) : 0;

  // Experts.
  const experts = n(config, "num_local_experts", "num_experts", "n_routed_experts") ?? 0;
  const topk = n(config, "num_experts_per_tok") ?? 0;
  const expertFf = n(config, "moe_intermediate_size") ?? ff;
  const sharedCount = deepseek ? (n(config, "n_shared_experts") ?? 0) : n(config, "shared_expert_intermediate_size") ? 1 : 0;
  const sharedFf = deepseek ? expertFf : (n(config, "shared_expert_intermediate_size") ?? 0);
  const denseFirst = deepseek ? (n(config, "first_k_dense_replace") ?? 0) : 0;
  const mlpOnly = Array.isArray(config.mlp_only_layers) ? (config.mlp_only_layers as number[]) : [];
  const sparseStep = n(config, "decoder_sparse_step") ?? 1;
  const isMoeLayer = (i: number) => experts > 0 && i >= denseFirst && !mlpOnly.includes(i) && (i + 1) % sparseStep === 0;

  // Latent attention.
  const kvRank = n(config, "kv_lora_rank");
  const latent = deepseek && kvRank !== undefined;

  const b = new Builder(repo || String((config.architectures as string[] | undefined)?.[0] ?? type), "language", repo ? `huggingface:${repo}` : "huggingface");
  let x = b.input(String(seqLen), "Tokens");
  x = b.node("embedding", x, { vocab, dim: d }, "Token embedding");
  if (gpt2) x = b.node("posembed", x, { max: context }, "Position embedding");

  const norm = (input: string, label: string) => (gpt2 ? b.node("layernorm", input, {}, label) : b.node("rmsnorm", input, {}, label));
  const attention = (input: string) =>
    latent
      ? b.node(
          "mla",
          input,
          {
            heads,
            q_rank: n(config, "q_lora_rank") ?? 0,
            kv_rank: kvRank!,
            qk_nope_dim: n(config, "qk_nope_head_dim")!,
            qk_rope_dim: n(config, "qk_rope_head_dim")!,
            v_dim: n(config, "v_head_dim")!,
          },
          "Latent attention",
        )
      : b.node("attention", input, { heads, kv, head_dim: headDim, qkv_bias: qkvBias, o_bias: oBias, qk_norm: qkNorm, ...(window ? { window } : {}), rope: !gpt2 }, kv === heads ? "Attention" : kv === 1 ? "Multi-query attention" : "Grouped-query attention");
  const feedForward = (input: string, moe: boolean) =>
    moe
      ? b.node(
          "moe",
          input,
          { experts, topk, ff: expertFf, shared: sharedCount, shared_ff: sharedFf, shared_gate: type === "qwen2_moe", router_bias: type === "deepseek_v3" },
          `Mixture of ${experts} experts`,
        )
      : b.node("mlp", input, { ff, act: mlpAct, bias: gpt2 }, "Feed-forward");

  // Consecutive layers with the same shape of block share one repeated group.
  const kinds: boolean[] = Array.from({ length: layers }, (_, i) => isMoeLayer(i));
  let start = 0;
  while (start < layers) {
    let end = start;
    while (end + 1 < layers && kinds[end + 1] === kinds[start]) end++;
    const count = end - start + 1;
    const moe = kinds[start];
    const label = experts > 0 ? `${moe ? "Expert" : "Dense"} decoder layer` : "Decoder layer";
    const input = x;
    x = b.group(
      label,
      // One kind for dense and expert layers, so a diff lines them up and reports the swap.
      "decoder",
      () => {
        let h = norm(input, sandwich ? "Pre-attention norm" : "Attention norm");
        h = attention(h);
        if (sandwich) h = norm(h, "Post-attention norm");
        const mid = b.node("add", [h, input], {}, "Residual");
        let f = norm(mid, sandwich ? "Pre-feed-forward norm" : "Feed-forward norm");
        f = feedForward(f, moe);
        if (sandwich) f = norm(f, "Post-feed-forward norm");
        return b.node("add", [f, mid], {}, "Residual");
      },
      count,
    );
    start = end + 1;
  }
  x = norm(x, "Final norm");
  b.node("lmhead", x, { vocab, tied }, tied ? "Output (tied to embedding)" : "Output");

  const attentionName = latent ? `Latent (MLA), ${heads} heads` : kv === heads ? `Multi-head, ${heads} heads` : kv === 1 ? `Multi-query, ${heads} heads` : `Grouped-query, ${heads} heads / ${kv} KV`;
  const meta: Model["meta"] = {
    family: type,
    layers,
    "hidden size": d,
    attention: attentionName,
    "head dim": latent ? `${n(config, "qk_nope_head_dim")}+${n(config, "qk_rope_head_dim")} / v ${n(config, "v_head_dim")}` : headDim,
    "feed-forward": experts > 0 ? `${experts} experts, top-${topk}, ${expertFf} wide${sharedCount ? `, ${sharedCount} shared` : ""}${denseFirst ? `; first ${denseFirst} dense` : ""}` : `${mlpAct === "gelu" ? "GELU" : mlpAct === "geglu" ? "GeGLU" : "SwiGLU"}, ${ff} wide`,
    vocab,
    context,
    positions: gpt2 ? "learned" : `RoPE θ ${n(config, "rope_theta") ?? "default"}`,
    norm: `${gpt2 ? "LayerNorm" : "RMSNorm"}, ${sandwich ? "before and after" : "before each part"}`,
    "tied embeddings": tied,
    "sliding window": typeof config.sliding_window === "number" && config.use_sliding_window !== false ? (config.sliding_window as number) : "none",
    "QK norm": qkNorm,
    "attention bias": qkvBias,
  };
  if (deepseek && n(config, "num_nextn_predict_layers")) meta["multi-token prediction"] = `${n(config, "num_nextn_predict_layers")} extra layer (not counted)`;
  if (!SUPPORTED_FAMILIES.includes(type)) meta.note = `"${type}" is read as a standard decoder; check it against the model's code.`;
  return b.build(meta, seqLen);
}
