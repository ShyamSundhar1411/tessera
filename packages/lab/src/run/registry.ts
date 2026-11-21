import imagenet from "./imagenet-labels.json";

// Pretrained ImageNet classifiers published as ONNX on Hugging Face. Every URL
// was checked by hand: huggingface.co answers the CORS request and redirects
// to its CDN, which sends Access-Control-Allow-Origin: *. Preprocessing comes
// from each repo's preprocessor_config.json, labels from its config.json
// (all share the standard ImageNet-1k order kept in imagenet-labels.json).

export interface InputSpec {
  /** The square side the model reads. */
  size: number;
  /** The shortest edge is resized to this before the centre crop. */
  resize: number;
  /** Per-channel mean and standard deviation after scaling pixels to 0..1, in RGB order. */
  mean: [number, number, number];
  std: [number, number, number];
  /** MobileViT was trained on BGR images. */
  channelOrder: "rgb" | "bgr";
}

export interface PretrainedEntry {
  id: string;
  name: string;
  /** The Hugging Face repo it comes from. */
  repo: string;
  url: string;
  /** Download size in megabytes. */
  sizeMB: number;
  /** "fp32" is the original weights; "uint8" is dynamically quantised, about a quarter the size and slightly less accurate. */
  precision: "fp32" | "uint8";
  /** The catalog model with the same architecture, so the UI can show its graph next to the predictions. */
  catalogId?: string;
  input: InputSpec;
  /** "imagenet-1k+background" means output 0 is an extra background class (Google's TensorFlow MobileNets). */
  labels: "imagenet-1k" | "imagenet-1k+background";
  note?: string;
}

const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [number, number, number] = [0.229, 0.224, 0.225];
const hf = (repo: string, file: string) => `https://huggingface.co/${repo}/resolve/main/${file}`;

// crop_pct 0.875 with a 224 crop means resizing the short edge to 256 first.
const imagenet224: InputSpec = { size: 224, resize: 256, mean: IMAGENET_MEAN, std: IMAGENET_STD, channelOrder: "rgb" };

export const PRETRAINED: PretrainedEntry[] = [
  {
    id: "resnet-18",
    name: "ResNet-18",
    repo: "Xenova/resnet-18",
    url: hf("Xenova/resnet-18", "onnx/model.onnx"),
    sizeMB: 46.8,
    precision: "fp32",
    catalogId: "resnet18",
    input: imagenet224,
    labels: "imagenet-1k",
  },
  {
    id: "resnet-50",
    name: "ResNet-50",
    repo: "Xenova/resnet-50",
    url: hf("Xenova/resnet-50", "onnx/model_quantized.onnx"),
    sizeMB: 25.8,
    precision: "uint8",
    catalogId: "resnet50",
    input: imagenet224,
    labels: "imagenet-1k",
    note: "The full-precision file (onnx/model.onnx) is 102.2 MB, so the quantised one is used.",
  },
  {
    id: "mobilenet-v2",
    name: "MobileNetV2",
    repo: "onnx-community/mobilenet_v2_1.0_224",
    url: hf("onnx-community/mobilenet_v2_1.0_224", "onnx/model.onnx"),
    sizeMB: 14.0,
    precision: "fp32",
    catalogId: "mobilenet-v2",
    input: { size: 224, resize: 256, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], channelOrder: "rgb" },
    labels: "imagenet-1k+background",
  },
  {
    id: "efficientnet-b0",
    name: "EfficientNet-B0",
    repo: "onnxmodelzoo/efficientnet_b0_Opset17",
    url: hf("onnxmodelzoo/efficientnet_b0_Opset17", "efficientnet_b0_Opset17.onnx"),
    sizeMB: 21.1,
    precision: "fp32",
    catalogId: "efficientnet-b0",
    // timm's efficientnet_b0 settings (crop_pct 0.875, ImageNet mean and std); the repo has no preprocessor_config.json.
    input: imagenet224,
    labels: "imagenet-1k",
    note: "timm's efficientnet_b0 exported by the ONNX model zoo; 5,288,548 parameters, the same as the catalog model.",
  },
  {
    id: "mobilenet-v4-small",
    name: "MobileNetV4 small",
    repo: "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    url: hf("onnx-community/mobilenetv4_conv_small.e2400_r224_in1k", "onnx/model.onnx"),
    sizeMB: 15.1,
    precision: "fp32",
    input: imagenet224,
    labels: "imagenet-1k",
  },
  {
    id: "mobilevit-small",
    name: "MobileViT small",
    repo: "Xenova/mobilevit-small",
    url: hf("Xenova/mobilevit-small", "onnx/model.onnx"),
    sizeMB: 22.6,
    precision: "fp32",
    // No normalisation, channels flipped to BGR, 288 then a 256 crop.
    input: { size: 256, resize: 288, mean: [0, 0, 0], std: [1, 1, 1], channelOrder: "bgr" },
    labels: "imagenet-1k",
  },
  {
    id: "convnext-tiny",
    name: "ConvNeXt tiny",
    repo: "Xenova/convnext-tiny-224",
    url: hf("Xenova/convnext-tiny-224", "onnx/model_quantized.onnx"),
    sizeMB: 29.3,
    precision: "uint8",
    input: imagenet224,
    labels: "imagenet-1k",
    note: "The full-precision file is 114.5 MB, so the quantised one is used.",
  },
];

export const IMAGENET_LABELS: string[] = imagenet;

export function labelsFor(entry: PretrainedEntry): string[] {
  return entry.labels === "imagenet-1k+background" ? ["background", ...IMAGENET_LABELS] : IMAGENET_LABELS;
}

export const pretrainedEntry = (id: string) => PRETRAINED.find((entry) => entry.id === id);
