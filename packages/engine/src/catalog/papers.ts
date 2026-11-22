import { Builder } from "../builder";
import type { Model } from "../types";

// Landmark image architectures from papers, built like torchvision's versions
// (or, for U-Net, the widely used padded implementation) so their parameter
// counts match.

const relu = (b: Builder, x: string) => b.node("relu", x);

/** AlexNet (Krizhevsky, Sutskever and Hinton, 2012): large early filters, ReLU and dropout, on GPUs. */
export function alexnet(): Model {
  const b = new Builder("AlexNet", "vision");
  let x = b.input("3x224x224", "Image");
  x = b.group("Features", "features", () => {
    let y = relu(b, b.node("conv2d", x, { out: 64, k: 11, s: 4, p: 2 }));
    y = b.node("maxpool", y, { k: 3, s: 2 });
    y = relu(b, b.node("conv2d", y, { out: 192, k: 5, p: 2 }));
    y = b.node("maxpool", y, { k: 3, s: 2 });
    y = relu(b, b.node("conv2d", y, { out: 384, k: 3 }));
    y = relu(b, b.node("conv2d", y, { out: 256, k: 3 }));
    y = relu(b, b.node("conv2d", y, { out: 256, k: 3 }));
    return b.node("maxpool", y, { k: 3, s: 2 });
  });
  x = b.group("Classifier", "classifier", () => {
    let y = b.node("flatten", x);
    y = b.node("dropout", y, { rate: 0.5 });
    y = relu(b, b.node("linear", y, { out: 4096 }));
    y = b.node("dropout", y, { rate: 0.5 });
    y = relu(b, b.node("linear", y, { out: 4096 }));
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "AlexNet", depth: 8, blocks: "plain conv, then dense", skips: "none", input: "224×224" });
}

/** SqueezeNet 1.0 (Iandola et al., 2016): AlexNet-level accuracy with 50x fewer parameters, from 1×1 "squeeze" layers. */
export function squeezenet(): Model {
  const b = new Builder("SqueezeNet 1.0", "vision");
  let x = b.input("3x224x224", "Image");
  x = relu(b, b.node("conv2d", x, { out: 96, k: 7, s: 2, p: 0 }));
  x = b.node("maxpool", x, { k: 3, s: 2, ceil: true });
  let index = 0;
  const fire = (squeeze: number, expand: number) => {
    index++;
    const input = x;
    x = b.group(`Fire ${index + 1}`, "fire", () => {
      const s = relu(b, b.node("conv2d", input, { out: squeeze, k: 1 }, "Squeeze"));
      const e1 = relu(b, b.node("conv2d", s, { out: expand, k: 1 }, "Expand 1×1"));
      const e3 = relu(b, b.node("conv2d", s, { out: expand, k: 3 }, "Expand 3×3"));
      return b.node("concat", [e1, e3], {}, "Join");
    });
  };
  fire(16, 64);
  fire(16, 64);
  fire(32, 128);
  x = b.node("maxpool", x, { k: 3, s: 2, ceil: true });
  fire(32, 128);
  fire(48, 192);
  fire(48, 192);
  fire(64, 256);
  x = b.node("maxpool", x, { k: 3, s: 2, ceil: true });
  fire(64, 256);
  x = b.group("Classifier", "classifier", () => {
    let y = b.node("dropout", x, { rate: 0.5 });
    y = relu(b, b.node("conv2d", y, { out: 1000, k: 1 }));
    y = b.node("gap", y);
    return b.node("flatten", y);
  });
  return b.build({ family: "SqueezeNet", depth: 18, blocks: "fire (squeeze 1×1, expand 1×1 and 3×3)", skips: "none", input: "224×224" });
}

const basicConv = (b: Builder, x: string, out: number, k: number, s = 1, p?: number) => {
  let y = b.node("conv2d", x, { out, k, s, ...(p !== undefined ? { p } : {}), bias: false });
  y = b.node("batchnorm", y);
  return relu(b, y);
};

/** GoogLeNet (Szegedy et al., 2014): Inception modules run 1×1, 3×3 and pooling branches side by side and join them. */
export function googlenet(): Model {
  const b = new Builder("GoogLeNet (Inception v1)", "vision");
  let x = b.input("3x224x224", "Image");
  x = b.group("Stem", "stem", () => {
    let y = basicConv(b, x, 64, 7, 2, 3);
    y = b.node("maxpool", y, { k: 3, s: 2, ceil: true });
    y = basicConv(b, y, 64, 1);
    y = basicConv(b, y, 192, 3, 1, 1);
    return b.node("maxpool", y, { k: 3, s: 2, ceil: true });
  });
  const inception = (name: string, c1: number, r3: number, c3: number, r5: number, c5: number, pool: number) => {
    const input = x;
    x = b.group(`Inception ${name}`, "inception", () => {
      const a = basicConv(b, input, c1, 1);
      const m = basicConv(b, basicConv(b, input, r3, 1), c3, 3);
      // torchvision's version uses 3×3 here where the paper says 5×5; kept so the count matches.
      const n = basicConv(b, basicConv(b, input, r5, 1), c5, 3);
      const p = basicConv(b, b.node("maxpool", input, { k: 3, s: 1, p: 1, ceil: true }), pool, 1);
      return b.node("concat", [a, m, n, p], {}, "Join branches");
    });
  };
  inception("3a", 64, 96, 128, 16, 32, 32);
  inception("3b", 128, 128, 192, 32, 96, 64);
  x = b.node("maxpool", x, { k: 3, s: 2, ceil: true });
  inception("4a", 192, 96, 208, 16, 48, 64);
  inception("4b", 160, 112, 224, 24, 64, 64);
  inception("4c", 128, 128, 256, 24, 64, 64);
  inception("4d", 112, 144, 288, 32, 64, 64);
  inception("4e", 256, 160, 320, 32, 128, 128);
  x = b.node("maxpool", x, { k: 2, s: 2, ceil: true });
  inception("5a", 256, 160, 320, 32, 128, 128);
  inception("5b", 384, 192, 384, 48, 128, 128);
  x = b.group("Classifier", "classifier", () => {
    let y = b.node("gap", x);
    y = b.node("flatten", y);
    y = b.node("dropout", y, { rate: 0.2 });
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "Inception", depth: 22, blocks: "inception (parallel 1×1, 3×3, pool branches)", skips: "none, branches join by concatenation", input: "224×224" });
}

/** DenseNet-121 (Huang et al., 2016): inside a block, every layer receives every earlier layer's output. */
export function densenet121(): Model {
  const b = new Builder("DenseNet-121", "vision");
  const growth = 32;
  let x = b.input("3x224x224", "Image");
  x = b.group("Stem", "stem", () => {
    let y = b.node("conv2d", x, { out: 64, k: 7, s: 2, p: 3, bias: false });
    y = relu(b, b.node("batchnorm", y));
    return b.node("maxpool", y, { k: 3, s: 2, p: 1 });
  });
  let channels = 64;
  [6, 12, 24, 16].forEach((layers, blockIndex) => {
    for (let i = 0; i < layers; i++) {
      const input = x;
      x = b.group(`Dense block ${blockIndex + 1}, layer ${i + 1}`, "dense-layer", () => {
        let y = relu(b, b.node("batchnorm", input));
        y = b.node("conv2d", y, { out: 4 * growth, k: 1, bias: false }, "Bottleneck 1×1");
        y = relu(b, b.node("batchnorm", y));
        y = b.node("conv2d", y, { out: growth, k: 3, bias: false }, `Adds ${growth} channels`);
        return b.node("concat", [input, y], {}, "Keep everything so far");
      });
      channels += growth;
    }
    if (blockIndex < 3) {
      const input = x;
      x = b.group(`Transition ${blockIndex + 1}`, "transition", () => {
        let y = relu(b, b.node("batchnorm", input));
        y = b.node("conv2d", y, { out: channels / 2, k: 1, bias: false });
        return b.node("avgpool", y, { k: 2 });
      });
      channels /= 2;
    }
  });
  x = b.group("Classifier", "classifier", () => {
    let y = relu(b, b.node("batchnorm", x));
    y = b.node("gap", y);
    y = b.node("flatten", y);
    return b.node("linear", y, { out: 1000 });
  });
  return b.build({ family: "DenseNet", depth: 121, blocks: "dense (each layer concatenates all earlier ones)", skips: "dense concatenation inside each block", input: "224×224" });
}

/** U-Net (Ronneberger, Fischer and Brox, 2015): an encoder that shrinks, a decoder that grows back, and skips that hand detail across. */
export function unet(): Model {
  const b = new Builder("U-Net", "vision");
  const double = (x: string, out: number) => {
    let y = b.node("conv2d", x, { out, k: 3, bias: false });
    y = relu(b, b.node("batchnorm", y));
    y = b.node("conv2d", y, { out, k: 3, bias: false });
    return relu(b, b.node("batchnorm", y));
  };
  let x = b.input("3x256x256", "Image");
  const skips: string[] = [];
  x = b.group("Encoder 1", "encoder", () => double(x, 64));
  skips.push(x);
  [128, 256, 512, 1024].forEach((out, i) => {
    const input = x;
    x = b.group(`Encoder ${i + 2}`, "encoder", () => double(b.node("maxpool", input, { k: 2 }), out));
    if (i < 3) skips.push(x);
  });
  [512, 256, 128, 64].forEach((out, i) => {
    const input = x;
    const skip = skips[skips.length - 1 - i];
    x = b.group(`Decoder ${i + 1}`, "decoder", () => {
      const up = b.node("upconv2d", input, { out, k: 2, s: 2 }, "Up 2×");
      const joined = b.node("concat", [skip, up], {}, "Skip across the U");
      return double(joined, out);
    });
  });
  b.node("conv2d", x, { out: 2, k: 1 }, "Per-pixel classes");
  return b.build({ family: "U-Net", depth: 23, blocks: "encoder and decoder of double 3×3 convs", skips: "copy-and-concatenate across the U", input: "256×256, output per pixel" });
}
