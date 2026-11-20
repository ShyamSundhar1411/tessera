import deepseekV3 from "../../configs/deepseek-ai_DeepSeek-V3.json";
import gemma2 from "../../configs/unsloth_gemma-2-9b.json";
import gemma3 from "../../configs/unsloth_gemma-3-1b-pt.json";
import smollm2 from "../../configs/HuggingFaceTB_SmolLM2-135M.json";
import phi3 from "../../configs/microsoft_Phi-3-mini-4k-instruct.json";
import mistral from "../../configs/mistralai_Mistral-7B-v0.1.json";
import mixtral from "../../configs/mistralai_Mixtral-8x7B-v0.1.json";
import gpt2 from "../../configs/openai-community_gpt2.json";
import qwen25 from "../../configs/Qwen_Qwen2.5-7B.json";
import qwen3Moe from "../../configs/Qwen_Qwen3-30B-A3B.json";
import qwen3 from "../../configs/Qwen_Qwen3-8B.json";
import llama31 from "../../configs/unsloth_Meta-Llama-3.1-8B.json";
import { fromHuggingFace } from "../import/huggingface";
import type { Model } from "../types";
import { cifarCnn, efficientnetB0, lenet5, mnistMlp, mobilenetV2, resnet18, resnet50, vgg16 } from "./vision";

export interface CatalogEntry {
  id: string;
  name: string;
  shelf: "Language models" | "Image models" | "Small models";
  blurb: string;
  /** Parameters as the authors publish them, for the tests and the catalog card. */
  published?: string;
  build: () => Model;
}

/** Real config.json files, downloaded from Hugging Face and kept with the app. */
const hf = (config: unknown, repo: string) => () => fromHuggingFace(config as Record<string, unknown>, repo);

export const CATALOG: CatalogEntry[] = [
  { id: "llama-3.1-8b", name: "Llama 3.1 8B", shelf: "Language models", blurb: "Meta's dense decoder: grouped-query attention, SwiGLU, RoPE.", published: "8.0B", build: hf(llama31, "meta-llama/Llama-3.1-8B") },
  { id: "mistral-7b", name: "Mistral 7B", shelf: "Language models", blurb: "Grouped-query attention with a sliding window.", published: "7.2B", build: hf(mistral, "mistralai/Mistral-7B-v0.1") },
  { id: "mixtral-8x7b", name: "Mixtral 8x7B", shelf: "Language models", blurb: "Mistral's layers with 8 experts, 2 used per token.", published: "46.7B total, 12.9B active", build: hf(mixtral, "mistralai/Mixtral-8x7B-v0.1") },
  { id: "qwen2.5-7b", name: "Qwen2.5 7B", shelf: "Language models", blurb: "Bias on queries, keys and values; a very large vocabulary.", published: "7.6B", build: hf(qwen25, "Qwen/Qwen2.5-7B") },
  { id: "qwen3-8b", name: "Qwen3 8B", shelf: "Language models", blurb: "Drops the bias, adds normalisation to queries and keys.", published: "8.2B", build: hf(qwen3, "Qwen/Qwen3-8B") },
  { id: "qwen3-30b-a3b", name: "Qwen3 30B-A3B", shelf: "Language models", blurb: "128 small experts, 8 used per token.", published: "30.5B total, 3.3B active", build: hf(qwen3Moe, "Qwen/Qwen3-30B-A3B") },
  { id: "deepseek-v3", name: "DeepSeek-V3", shelf: "Language models", blurb: "Latent attention and 256 fine-grained experts with a shared one.", published: "671B total, 37B active", build: hf(deepseekV3, "deepseek-ai/DeepSeek-V3") },
  { id: "gemma-2-9b", name: "Gemma 2 9B", shelf: "Language models", blurb: "Norms before and after each part, GeGLU, tied embeddings.", published: "9.2B", build: hf(gemma2, "google/gemma-2-9b") },
  { id: "gemma-3-1b", name: "Gemma 3 1B", shelf: "Language models", blurb: "One key/value head shared by all, a 262k vocabulary.", published: "1.0B", build: hf(gemma3, "google/gemma-3-1b-pt") },
  { id: "phi-3-mini", name: "Phi-3 mini", shelf: "Language models", blurb: "A small model with full multi-head attention.", published: "3.8B", build: hf(phi3, "microsoft/Phi-3-mini-4k-instruct") },
  { id: "smollm2-135m", name: "SmolLM2 135M", shelf: "Language models", blurb: "A tiny Llama-style model, small enough to study whole.", published: "135M", build: hf(smollm2, "HuggingFaceTB/SmolLM2-135M") },
  { id: "gpt2", name: "GPT-2", shelf: "Language models", blurb: "The 2019 original: learned positions, LayerNorm, GELU, full attention.", published: "124M", build: hf(gpt2, "openai-community/gpt2") },
  { id: "vgg16", name: "VGG16", shelf: "Image models", blurb: "Thirteen 3×3 convolutions in a straight line, then big dense layers.", published: "138.4M", build: vgg16 },
  { id: "resnet18", name: "ResNet-18", shelf: "Image models", blurb: "Skip connections around every pair of convolutions.", published: "11.7M", build: resnet18 },
  { id: "resnet50", name: "ResNet-50", shelf: "Image models", blurb: "Bottleneck blocks: squeeze to 1×1, 3×3, expand again.", published: "25.6M", build: resnet50 },
  { id: "mobilenet-v2", name: "MobileNetV2", shelf: "Image models", blurb: "Cheap depthwise convolutions in inverted residual blocks.", published: "3.5M", build: mobilenetV2 },
  { id: "efficientnet-b0", name: "EfficientNet-B0", shelf: "Image models", blurb: "MBConv blocks with squeeze-and-excitation gates.", published: "5.3M", build: efficientnetB0 },
  { id: "mnist-mlp", name: "MNIST MLP", shelf: "Small models", blurb: "Three dense layers for handwritten digits. Trains in seconds.", build: mnistMlp },
  { id: "lenet5", name: "LeNet-5", shelf: "Small models", blurb: "The 1998 convolutional network for digits.", build: lenet5 },
  { id: "cifar-cnn", name: "Small CNN", shelf: "Small models", blurb: "A compact CIFAR-10 network, ready for experiments.", build: () => cifarCnn(false) },
  { id: "cifar-cnn-skip", name: "Small CNN with a skip", shelf: "Small models", blurb: "The same network with one residual connection added.", build: () => cifarCnn(true) },
];

export const catalogModel = (id: string) => CATALOG.find((entry) => entry.id === id)?.build();
