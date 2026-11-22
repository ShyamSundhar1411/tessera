import { catalogModel, type Model } from "tessera-engine";
import { buildGraph, type GraphModel } from "../train/graph";
import { assignWeights, fetchCached, parseSafetensors, type SafeTensor } from "./gpt2";
import { IMAGENET_LABELS } from "./registry";

// torchvision's trained ResNets in the tf.js graph, so the trace view shows
// what a real ImageNet model sees at every layer. timm republishes
// torchvision's weights as safetensors with torchvision's names.

export interface VisionWeights {
  catalogId: string;
  name: string;
  url: string;
  bytes: number;
}

export const VISION_WEIGHTS: VisionWeights[] = [
  { catalogId: "resnet18", name: "ResNet-18", url: "https://huggingface.co/timm/resnet18.tv_in1k/resolve/main/model.safetensors", bytes: 46_807_446 },
  { catalogId: "resnet50", name: "ResNet-50", url: "https://huggingface.co/timm/resnet50.tv_in1k/resolve/main/model.safetensors", bytes: 102_469_840 },
];

export const IMAGENET_NORMALIZE = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

/**
 * torchvision module names in the order the modules are defined, which is the
 * order the catalog builds layers in: stem conv1 and bn1, then each block's
 * conv1, bn1, conv2, bn2 (conv3, bn3), downsample, then fc.
 */
function resnetOrder(a: string, b: string) {
  const key = (name: string) =>
    name
      .split(".")
      .map((part) => {
        const layer = /^(conv|bn)(\d)$/.exec(part);
        if (layer) return `a${layer[2]}${layer[1] === "conv" ? 0 : 1}`;
        if (part === "downsample") return "b";
        if (part === "fc") return "z";
        if (/^layer\d$/.test(part)) return `y${part.slice(5)}`;
        return part.padStart(4, "0");
      })
      .join(".");
  return key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0;
}

/** PyTorch [out, in, kh, kw] to tf's [kh, kw, in, out]. */
function convToTf(t: SafeTensor): SafeTensor {
  const [o, i, h, w] = t.shape;
  const data = new Float32Array(t.data.length);
  for (let a = 0; a < o; a++) for (let b = 0; b < i; b++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[((y * w + x) * i + b) * o + a] = t.data[((a * i + b) * h + y) * w + x];
  return { shape: [h, w, i, o], data };
}

/** PyTorch's linear [out, in] to the graph's [in, out]. */
function transpose(t: SafeTensor): SafeTensor {
  const [o, i] = t.shape;
  const data = new Float32Array(t.data.length);
  for (let a = 0; a < o; a++) for (let b = 0; b < i; b++) data[b * o + a] = t.data[a * i + b];
  return { shape: [i, o], data };
}

/** Graph variable (step key and part) for each torchvision tensor, pairing layers in definition order. */
export function resnetAssignments(model: Model, tensors: Map<string, SafeTensor>): Map<string, SafeTensor> {
  const modules = [...new Set([...tensors.keys()].map((name) => name.replace(/\.(weight|bias|running_mean|running_var)$/, "")))].sort(resnetOrder);
  const layers = model.nodes.filter((node) => node.op === "conv2d" || node.op === "batchnorm" || node.op === "linear");
  if (modules.length !== layers.length) throw new Error(`the weights have ${modules.length} layers but ${model.name} has ${layers.length}`);
  const out = new Map<string, SafeTensor>();
  const get = (name: string) => {
    const tensor = tensors.get(name);
    if (!tensor) throw new Error(`the weights have no ${name}`);
    return tensor;
  };
  layers.forEach((node, index) => {
    const name = modules[index];
    if (node.op === "conv2d") {
      out.set(`${node.id}/w`, convToTf(get(`${name}.weight`)));
      if (tensors.has(`${name}.bias`)) out.set(`${node.id}/_bias`, get(`${name}.bias`));
    } else if (node.op === "batchnorm") {
      out.set(`${node.id}/_gamma`, get(`${name}.weight`));
      out.set(`${node.id}/_beta`, get(`${name}.bias`));
      out.set(`${node.id}/running_mean`, get(`${name}.running_mean`));
      out.set(`${node.id}/running_var`, get(`${name}.running_var`));
    } else {
      out.set(`${node.id}/w`, transpose(get(`${name}.weight`)));
      out.set(`${node.id}/_bias`, get(`${name}.bias`));
    }
  });
  return out;
}

export interface LoadedVision {
  graph: GraphModel;
  model: Model;
  labels: string[];
}

/** Downloads (once) and builds a torchvision ResNet with its ImageNet weights. */
export async function loadVisionWeights(catalogId: string, onProgress?: (loaded: number, total: number) => void): Promise<LoadedVision> {
  const entry = VISION_WEIGHTS.find((item) => item.catalogId === catalogId);
  if (!entry) throw new Error(`no published weights for ${catalogId}`);
  const response = await fetchCached(entry.url, (loaded, total) => onProgress?.(loaded, total || entry.bytes));
  const tensors = parseSafetensors(await response.arrayBuffer());
  const model = catalogModel(catalogId)!;
  const graph = buildGraph(model, { seed: 0 });
  try {
    assignWeights(graph, resnetAssignments(model, tensors));
  } catch (error) {
    graph.dispose();
    throw error;
  }
  return { graph, model, labels: IMAGENET_LABELS };
}
