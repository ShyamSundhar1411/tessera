import { Builder } from "../builder";
import type { Model } from "../types";

// Transformers that are not decoder-only language models: the original
// encoder-decoder from "Attention Is All You Need", and the Vision
// Transformer. Encoders (BERT and family) come from their config.json.

/**
 * The base model from Vaswani et al. (2017): 6 encoder and 6 decoder layers,
 * width 512, 8 heads, a 2048-wide ReLU feed-forward, normalisation after each
 * part, sine-wave positions, and one 37,000-token table shared by the source
 * embedding, the target embedding and the output layer. The paper gives about
 * 65M parameters.
 */
export function transformerBase(): Model {
  const d = 512;
  const heads = 8;
  const ff = 2048;
  const vocab = 37000;
  const b = new Builder("Transformer (Attention Is All You Need)", "language");
  const src = b.input("128", "Source tokens");
  const tgt = b.input("128", "Target tokens");
  let x = b.node("embedding", src, { vocab, dim: d }, "Source embedding");
  x = b.node("posembed", x, { fixed: true }, "Sine positions");
  const encoderInput = x;
  const memory = b.group(
    "Encoder layer",
    "encoder",
    () => {
      let h = b.node("attention", encoderInput, { heads, causal: false, qkv_bias: true, o_bias: true, rope: false }, "Self-attention");
      h = b.node("add", [h, encoderInput], {}, "Residual");
      const mid = b.node("layernorm", h, {}, "Norm");
      let f = b.node("mlp", mid, { ff, act: "relu", bias: true }, "Feed-forward");
      f = b.node("add", [f, mid], {}, "Residual");
      return b.node("layernorm", f, {}, "Norm");
    },
    6,
  );
  let y = b.node("embedding", tgt, { vocab, dim: d, tied: true }, "Target embedding (shared)");
  y = b.node("posembed", y, { fixed: true }, "Sine positions");
  const decoderInput = y;
  y = b.group(
    "Decoder layer",
    "decoder",
    () => {
      let h = b.node("attention", decoderInput, { heads, causal: true, qkv_bias: true, o_bias: true, rope: false }, "Masked self-attention");
      h = b.node("add", [h, decoderInput], {}, "Residual");
      const first = b.node("layernorm", h, {}, "Norm");
      let c = b.node("cross_attention", [first, memory], { heads, bias: true }, "Cross-attention to the encoder");
      c = b.node("add", [c, first], {}, "Residual");
      const second = b.node("layernorm", c, {}, "Norm");
      let f = b.node("mlp", second, { ff, act: "relu", bias: true }, "Feed-forward");
      f = b.node("add", [f, second], {}, "Residual");
      return b.node("layernorm", f, {}, "Norm");
    },
    6,
  );
  b.node("lmhead", y, { vocab, tied: true }, "Output (shared table)");
  return b.build(
    {
      family: "transformer",
      layers: "6 encoder + 6 decoder",
      "hidden size": d,
      attention: `Multi-head, ${heads} heads; decoder adds cross-attention`,
      "feed-forward": `ReLU, ${ff} wide`,
      vocab,
      positions: "fixed sine waves",
      norm: "LayerNorm, after each part",
      "tied embeddings": "source, target and output share one table",
    },
    128,
  );
}

/**
 * ViT-B/16 (Dosovitskiy et al., 2020): the image is cut into 16x16 patches by
 * a strided convolution, the patches become a sequence with a class token
 * and learned positions, and a 12-layer encoder reads it. Built like
 * torchvision's vit_b_16, so its count matches exactly.
 */
export function vitB16(): Model {
  const d = 768;
  const b = new Builder("ViT-B/16", "vision");
  let x = b.input("3x224x224", "Image");
  x = b.node("conv2d", x, { out: d, k: 16, s: 16, p: 0 }, "Patches (16×16)");
  x = b.node("to_sequence", x, {}, "Patches as tokens");
  x = b.node("cls_token", x, {}, "Class token");
  x = b.node("posembed", x, { max: 197 }, "Positions");
  const input = x;
  x = b.group(
    "Encoder layer",
    "encoder",
    () => {
      let h = b.node("layernorm", input, {}, "Norm");
      h = b.node("attention", h, { heads: 12, causal: false, qkv_bias: true, o_bias: true, rope: false }, "Self-attention");
      const mid = b.node("add", [h, input], {}, "Residual");
      let f = b.node("layernorm", mid, {}, "Norm");
      f = b.node("mlp", f, { ff: 3072, act: "gelu", bias: true }, "Feed-forward");
      return b.node("add", [f, mid], {}, "Residual");
    },
    12,
  );
  x = b.node("layernorm", x, {}, "Final norm");
  x = b.node("take_first", x, {}, "Class token out");
  b.node("linear", x, { out: 1000 }, "Classifier");
  return b.build({ family: "ViT", depth: 12, blocks: "transformer encoder on 196 patches", skips: "residual around attention and feed-forward", input: "224×224" });
}
