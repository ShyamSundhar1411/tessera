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
import bert from "../../configs/google-bert_bert-base-uncased.json";
import roberta from "../../configs/FacebookAI_roberta-base.json";
import distilbert from "../../configs/distilbert_distilbert-base-uncased.json";
import { fromHuggingFace } from "../import/huggingface";
import type { Model } from "../types";
import { alexnet, densenet121, googlenet, squeezenet, unet } from "./papers";
import { transformerBase, vitB16 } from "./transformers";
import { cifarCnn, efficientnetB0, lenet5, mnistMlp, mobilenetV2, resnet18, resnet50, vgg16 } from "./vision";

export interface Paper {
  title: string;
  authors: string;
  year: number;
  url: string;
  /** The connection idea, in one sentence. */
  idea: string;
}

export interface CatalogEntry {
  id: string;
  name: string;
  shelf: "Language models" | "Encoders and seq2seq" | "Image models" | "Small models";
  blurb: string;
  /** Parameters as the authors publish them, for the tests and the catalog card. */
  published?: string;
  /** The paper that introduced it, and the change to the wiring it is remembered for. */
  paper?: Paper;
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
  { id: "transformer-base", name: "Transformer (2017)", shelf: "Encoders and seq2seq", blurb: "The original from Attention Is All You Need: an encoder, a decoder and cross-attention between them.", published: "65M", build: transformerBase },
  { id: "bert-base", name: "BERT base", shelf: "Encoders and seq2seq", blurb: "A bidirectional encoder: every token sees every other. Read from its config.json.", published: "110M", build: hf(bert, "google-bert/bert-base-uncased") },
  { id: "roberta-base", name: "RoBERTa base", shelf: "Encoders and seq2seq", blurb: "BERT's layers with a bigger vocabulary and better training.", published: "125M", build: hf(roberta, "FacebookAI/roberta-base") },
  { id: "distilbert-base", name: "DistilBERT", shelf: "Encoders and seq2seq", blurb: "BERT distilled to half the layers, no pooler.", published: "66M", build: hf(distilbert, "distilbert/distilbert-base-uncased") },
  { id: "vit-b16", name: "ViT-B/16", shelf: "Image models", blurb: "A transformer that reads an image as a sequence of 16×16 patches.", published: "86.6M", build: vitB16 },
  { id: "alexnet", name: "AlexNet", shelf: "Image models", blurb: "Big early filters, ReLU and dropout, trained on two GPUs. Started the deep learning wave.", published: "61.1M", build: alexnet },
  { id: "squeezenet", name: "SqueezeNet", shelf: "Image models", blurb: "AlexNet's accuracy with 50 times fewer parameters.", published: "1.25M", build: squeezenet },
  { id: "googlenet", name: "GoogLeNet (Inception)", shelf: "Image models", blurb: "Parallel 1×1, 3×3 and pooling branches, joined side by side.", published: "6.6M", build: googlenet },
  { id: "densenet121", name: "DenseNet-121", shelf: "Image models", blurb: "Every layer in a block receives every earlier layer's output.", published: "8.0M", build: densenet121 },
  { id: "unet", name: "U-Net", shelf: "Image models", blurb: "Shrink, grow back, and copy detail across the U. Labels every pixel.", published: "31.0M", build: unet },
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

const PAPERS: Record<string, Paper> = {
  "alexnet": { title: "ImageNet Classification with Deep Convolutional Neural Networks", authors: "Krizhevsky, Sutskever, Hinton", year: 2012, url: "https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html", idea: "Deep stacks of convolutions with ReLU and dropout, trained end to end on GPUs." },
  "vgg16": { title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", authors: "Simonyan, Zisserman", year: 2014, url: "https://arxiv.org/abs/1409.1556", idea: "Only 3\u00d73 convolutions, stacked deep: two 3\u00d73 layers see as much as one 5\u00d75 with fewer weights." },
  "googlenet": { title: "Going Deeper with Convolutions", authors: "Szegedy et al.", year: 2014, url: "https://arxiv.org/abs/1409.4842", idea: "Run several filter sizes side by side and concatenate, with 1\u00d71 convolutions keeping it cheap." },
  "resnet18": { title: "Deep Residual Learning for Image Recognition", authors: "He, Zhang, Ren, Sun", year: 2015, url: "https://arxiv.org/abs/1512.03385", idea: "Add each block's input to its output, so very deep networks still train: the skip connection." },
  "resnet50": { title: "Deep Residual Learning for Image Recognition", authors: "He, Zhang, Ren, Sun", year: 2015, url: "https://arxiv.org/abs/1512.03385", idea: "Bottleneck blocks (1\u00d71, 3\u00d73, 1\u00d71) with a skip around each make 50+ layers trainable." },
  "unet": { title: "U-Net: Convolutional Networks for Biomedical Image Segmentation", authors: "Ronneberger, Fischer, Brox", year: 2015, url: "https://arxiv.org/abs/1505.04597", idea: "Connect each encoder level to the matching decoder level, so fine detail skips the bottleneck." },
  "squeezenet": { title: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters", authors: "Iandola et al.", year: 2016, url: "https://arxiv.org/abs/1602.07360", idea: "Squeeze channels with 1\u00d71 convolutions before the expensive 3\u00d73 ones." },
  "densenet121": { title: "Densely Connected Convolutional Networks", authors: "Huang, Liu, van der Maaten, Weinberger", year: 2016, url: "https://arxiv.org/abs/1608.06993", idea: "Connect every layer to every later layer in a block by concatenation." },
  "transformer-base": { title: "Attention Is All You Need", authors: "Vaswani et al.", year: 2017, url: "https://arxiv.org/abs/1706.03762", idea: "Replace recurrence with attention: an encoder, a decoder, and cross-attention between them." },
  "mobilenet-v2": { title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks", authors: "Sandler et al.", year: 2018, url: "https://arxiv.org/abs/1801.04381", idea: "Expand, filter each channel separately, project back down, with the skip between the thin ends." },
  "bert-base": { title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", authors: "Devlin, Chang, Lee, Toutanova", year: 2018, url: "https://arxiv.org/abs/1810.04805", idea: "Let attention look both ways and learn by filling in masked words." },
  "gpt2": { title: "Language Models are Unsupervised Multitask Learners", authors: "Radford et al.", year: 2019, url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", idea: "A decoder-only transformer, scaled up and trained to predict the next token." },
  "roberta-base": { title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach", authors: "Liu et al.", year: 2019, url: "https://arxiv.org/abs/1907.11692", idea: "BERT's wiring unchanged; better training and a bigger vocabulary." },
  "distilbert-base": { title: "DistilBERT, a distilled version of BERT", authors: "Sanh, Debut, Chaumond, Wolf", year: 2019, url: "https://arxiv.org/abs/1910.01108", idea: "Half the layers, trained to imitate the full model." },
  "efficientnet-b0": { title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", authors: "Tan, Le", year: 2019, url: "https://arxiv.org/abs/1905.11946", idea: "Scale depth, width and resolution together, with squeeze-and-excitation gates in each block." },
  "vit-b16": { title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", authors: "Dosovitskiy et al.", year: 2020, url: "https://arxiv.org/abs/2010.11929", idea: "Cut the image into patches and treat them as tokens for a plain transformer encoder." },
  "mixtral-8x7b": { title: "Mixtral of Experts", authors: "Jiang et al.", year: 2024, url: "https://arxiv.org/abs/2401.04088", idea: "Replace each feed-forward with 8 experts and route every token to 2 of them." },
  "llama-3.1-8b": { title: "The Llama 3 Herd of Models", authors: "Llama Team, Meta", year: 2024, url: "https://arxiv.org/abs/2407.21783", idea: "A dense decoder with grouped-query attention: 4 query heads share each key/value head." },
  "mistral-7b": { title: "Mistral 7B", authors: "Jiang et al.", year: 2023, url: "https://arxiv.org/abs/2310.06825", idea: "Grouped-query attention plus a sliding window, so each token attends to a fixed span." },
  "deepseek-v3": { title: "DeepSeek-V3 Technical Report", authors: "DeepSeek-AI", year: 2024, url: "https://arxiv.org/abs/2412.19437", idea: "Squeeze keys and values through a small latent (MLA) and use 256 fine-grained experts with a shared one." },
  "qwen3-8b": { title: "Qwen3 Technical Report", authors: "Qwen Team", year: 2025, url: "https://arxiv.org/abs/2505.09388", idea: "Normalise queries and keys inside attention and drop the attention bias." },
  "gemma-2-9b": { title: "Gemma 2: Improving Open Language Models at a Practical Size", authors: "Gemma Team", year: 2024, url: "https://arxiv.org/abs/2408.00118", idea: "Norms both before and after each part, and alternating local and global attention." },
};

for (const entry of CATALOG) if (PAPERS[entry.id]) entry.paper = PAPERS[entry.id];

/** The catalog entries that come from a paper, oldest first: a timeline of connection ideas. */
export const PAPER_TIMELINE = CATALOG.filter((entry) => entry.paper).sort((a, b) => a.paper!.year - b.paper!.year || a.name.localeCompare(b.name));

export const catalogModel = (id: string) => CATALOG.find((entry) => entry.id === id)?.build();
