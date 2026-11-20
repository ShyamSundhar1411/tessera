import { Builder } from "../builder";
import type { Model } from "../types";

// Classic image models, built layer for layer like torchvision's versions so
// their parameter counts match the published ones exactly.

const convBn = (b: Builder, x: string, out: number, k: number, s = 1, act: string | null = "relu", groups?: number | "depthwise") => {
  let y = b.node("conv2d", x, { out, k, s, bias: false, ...(groups ? { groups } : {}) });
  y = b.node("batchnorm", y);
  return act ? b.node(act, y) : y;
};

export function vgg16(): Model {
  const b = new Builder("VGG16", "vision");
  let x = b.input("3x224x224", "Image");
  const stages: number[][] = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]];
  stages.forEach((widths, i) => {
    x = b.group(`Stage ${i + 1}`, "vgg-stage", () => {
      let y = x;
      for (const out of widths) {
        y = b.node("conv2d", y, { out, k: 3 });
        y = b.node("relu", y);
      }
      return b.node("maxpool", y, { k: 2 });
    });
  });
  x = b.group("Classifier", "classifier", () => {
    let y = b.node("flatten", x);
    y = b.node("linear", y, { out: 4096 });
    y = b.node("relu", y);
    y = b.node("dropout", y, { rate: 0.5 });
    y = b.node("linear", y, { out: 4096 });
    y = b.node("relu", y);
    y = b.node("dropout", y, { rate: 0.5 });
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "VGG", depth: 16, blocks: "plain conv stacks", skips: "none", input: "224×224" });
}

function resnet(name: string, layers: number[], bottleneck: boolean): Model {
  const b = new Builder(name, "vision");
  let x = b.input("3x224x224", "Image");
  x = b.group("Stem", "stem", () => {
    const y = convBn(b, x, 64, 7, 2);
    return b.node("maxpool", y, { k: 3, s: 2, p: 1 });
  });
  let channels = 64;
  const expansion = bottleneck ? 4 : 1;
  layers.forEach((count, stage) => {
    const width = 64 * 2 ** stage;
    for (let i = 0; i < count; i++) {
      const stride = stage > 0 && i === 0 ? 2 : 1;
      const out = width * expansion;
      x = b.group(`Stage ${stage + 1}, block ${i + 1}`, bottleneck ? "bottleneck" : "basic", () => {
        let y: string;
        if (bottleneck) {
          y = convBn(b, x, width, 1);
          y = convBn(b, y, width, 3, stride);
          y = convBn(b, y, out, 1, 1, null);
        } else {
          y = convBn(b, x, width, 3, stride);
          y = convBn(b, y, width, 3, 1, null);
        }
        const shortcut = stride !== 1 || channels !== out ? convBn(b, x, out, 1, stride, null) : x;
        const sum = b.node("add", [y, shortcut], {}, "Skip join");
        return b.node("relu", sum);
      });
      channels = out;
    }
  });
  x = b.group("Classifier", "classifier", () => {
    let y = b.node("gap", x);
    y = b.node("flatten", y);
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "ResNet", depth: name.replace(/\D/g, ""), blocks: bottleneck ? "bottleneck (1×1, 3×3, 1×1)" : "basic (3×3, 3×3)", skips: "residual around every block", input: "224×224" });
}

export const resnet18 = () => resnet("ResNet-18", [2, 2, 2, 2], false);
export const resnet50 = () => resnet("ResNet-50", [3, 4, 6, 3], true);

export function mobilenetV2(): Model {
  const b = new Builder("MobileNetV2", "vision");
  let x = b.input("3x224x224", "Image");
  x = b.group("Stem", "stem", () => convBn(b, x, 32, 3, 2, "relu6"));
  let channels = 32;
  const settings = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ];
  let index = 0;
  for (const [t, c, n, s] of settings) {
    for (let i = 0; i < n; i++) {
      const stride = i === 0 ? s : 1;
      index++;
      x = b.group(`Inverted residual ${index}`, "inverted-residual", () => {
        const hidden = channels * t;
        let y = x;
        if (t !== 1) y = convBn(b, y, hidden, 1, 1, "relu6");
        y = convBn(b, y, hidden, 3, stride, "relu6", "depthwise");
        y = convBn(b, y, c, 1, 1, null);
        return stride === 1 && channels === c ? b.node("add", [y, x], {}, "Skip join") : y;
      });
      channels = c;
    }
  }
  x = b.group("Classifier", "classifier", () => {
    let y = convBn(b, x, 1280, 1, 1, "relu6");
    y = b.node("gap", y);
    y = b.node("flatten", y);
    y = b.node("dropout", y, { rate: 0.2 });
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "MobileNet", depth: 53, blocks: "inverted residual (expand, depthwise, project)", skips: "residual when shape is kept", input: "224×224" });
}

export function efficientnetB0(): Model {
  const b = new Builder("EfficientNet-B0", "vision");
  let x = b.input("3x224x224", "Image");
  x = b.group("Stem", "stem", () => convBn(b, x, 32, 3, 2, "silu"));
  let channels = 32;
  const settings = [
    [1, 3, 1, 16, 1],
    [6, 3, 2, 24, 2],
    [6, 5, 2, 40, 2],
    [6, 3, 2, 80, 3],
    [6, 5, 1, 112, 3],
    [6, 5, 2, 192, 4],
    [6, 3, 1, 320, 1],
  ];
  let index = 0;
  for (const [expand, k, s, out, n] of settings) {
    for (let i = 0; i < n; i++) {
      const stride = i === 0 ? s : 1;
      index++;
      x = b.group(`MBConv ${index}`, "mbconv", () => {
        const hidden = channels * expand;
        let y = x;
        if (expand !== 1) y = convBn(b, y, hidden, 1, 1, "silu");
        y = convBn(b, y, hidden, k, stride, "silu", "depthwise");
        // Squeeze and excitation: a small gate that reweights the channels.
        const squeezed = Math.max(1, Math.floor(channels / 4));
        let gate = b.node("gap", y, {}, "Squeeze");
        gate = b.node("conv2d", gate, { out: squeezed, k: 1 });
        gate = b.node("silu", gate);
        gate = b.node("conv2d", gate, { out: hidden, k: 1 });
        gate = b.node("sigmoid", gate);
        y = b.node("mul", [y, gate], {}, "Excite");
        y = convBn(b, y, out, 1, 1, null);
        return stride === 1 && channels === out ? b.node("add", [y, x], {}, "Skip join") : y;
      });
      channels = out;
    }
  }
  x = b.group("Classifier", "classifier", () => {
    let y = convBn(b, x, 1280, 1, 1, "silu");
    y = b.node("gap", y);
    y = b.node("flatten", y);
    y = b.node("dropout", y, { rate: 0.2 });
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "EfficientNet", depth: 82, blocks: "MBConv with squeeze-and-excitation", skips: "residual when shape is kept", input: "224×224" });
}

// -- Small models for learning and for training in the browser ----------------

export function mnistMlp(): Model {
  const b = new Builder("MNIST MLP", "vision");
  let x = b.input("1x28x28", "Digit");
  x = b.node("flatten", x);
  x = b.node("linear", x, { out: 128 });
  x = b.node("relu", x);
  x = b.node("linear", x, { out: 64 });
  x = b.node("relu", x);
  b.node("linear", x, { out: 10 });
  return b.build({ family: "MLP", depth: 3, blocks: "dense layers", skips: "none", input: "28×28 grey" });
}

export function lenet5(): Model {
  const b = new Builder("LeNet-5", "vision");
  let x = b.input("1x28x28", "Digit");
  x = b.node("conv2d", x, { out: 6, k: 5, p: 2 });
  x = b.node("tanh", x);
  x = b.node("avgpool", x, { k: 2 });
  x = b.node("conv2d", x, { out: 16, k: 5, p: 0 });
  x = b.node("tanh", x);
  x = b.node("avgpool", x, { k: 2 });
  x = b.node("flatten", x);
  x = b.node("linear", x, { out: 120 });
  x = b.node("tanh", x);
  x = b.node("linear", x, { out: 84 });
  x = b.node("tanh", x);
  b.node("linear", x, { out: 10 });
  return b.build({ family: "LeNet", depth: 5, blocks: "conv, pool", skips: "none", input: "28×28 grey" });
}

/** A small CIFAR-10 network with one optional shortcut, the starting point for skip connection experiments. */
export function cifarCnn(withSkip = false): Model {
  const b = new Builder(withSkip ? "Small CNN with a skip" : "Small CNN", "vision");
  let x = b.input("3x32x32", "Image");
  x = convBn(b, x, 32, 3);
  x = b.group("Block", "plain", () => {
    let y = convBn(b, x, 32, 3);
    y = convBn(b, y, 32, 3, 1, withSkip ? null : "relu");
    return withSkip ? b.node("relu", b.node("add", [y, x], {}, "Skip join")) : y;
  });
  x = b.node("maxpool", x, { k: 2 });
  x = convBn(b, x, 64, 3);
  x = b.node("maxpool", x, { k: 2 });
  x = b.node("gap", x);
  x = b.node("flatten", x);
  b.node("linear", x, { out: 10 });
  return b.build({ family: "Small CNN", depth: 5, blocks: "conv, batch norm, relu", skips: withSkip ? "one residual" : "none", input: "32×32 colour" });
}
