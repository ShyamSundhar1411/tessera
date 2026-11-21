import { describe, expect, test } from "bun:test";
import * as tf from "@tensorflow/tfjs";
import { analyze, Builder, catalogModel } from "tessera-engine";
import { buildTf, toTf, trainableParams, trainingProblem } from "./toTf";

await tf.setBackend("cpu");

// The strongest check that the translation is right: tf.js must hold exactly
// as many trainable parameters as the engine counts (which match torchvision).
describe("toTf matches the engine's parameter counts", () => {
  const expected: [string, number[]][] = [
    ["mnist-mlp", [10]],
    ["lenet5", [10]],
    ["cifar-cnn", [10]],
    ["cifar-cnn-skip", [10]],
    ["efficientnet-b0", [1000]],
    ["mobilenet-v2", [1000]],
    ["resnet18", [1000]],
  ];
  for (const [id, output] of expected) {
    test(id, () => {
      const model = catalogModel(id)!;
      const tfModel = toTf(model);
      expect(trainableParams(tfModel)).toBe(analyze(model).params);
      expect(tfModel.outputs[0].shape.slice(1)).toEqual(output);
      tfModel.dispose();
    });
  }
});

describe("toTf", () => {
  test("refuses language model ops and points to the Colab export", () => {
    const message = trainingProblem(catalogModel("smollm2-135m")!);
    expect(message).toContain("embedding");
    expect(message).toContain("Colab");
  });

  test("refuses grouped convolutions that are not depthwise", () => {
    const b = new Builder("Grouped", "vision");
    let x = b.input("8x16x16");
    x = b.node("conv2d", x, { out: 8, k: 3, groups: 2 });
    x = b.node("gap", x);
    x = b.node("flatten", x);
    b.node("linear", x, { out: 4 });
    expect(trainingProblem(b.build())).toContain("not supported for training in the browser");
  });

  test("unrolls repeated blocks", () => {
    const b = new Builder("Repeated", "vision");
    let x = b.input("1x8x8");
    x = b.node("conv2d", x, { out: 4, k: 3 });
    x = b.group("Block", "plain", () => {
      const y = b.node("conv2d", x, { out: 4, k: 3 });
      return b.node("add", [b.node("relu", y), x]);
    }, 3);
    x = b.node("flatten", x);
    b.node("linear", x, { out: 2 });
    const model = b.build();
    const built = buildTf(model);
    expect(trainableParams(built.model)).toBe(analyze(model).params);
    const convs = built.model.layers.filter((layer) => layer.getClassName() === "Conv2D");
    expect(convs.length).toBe(4);
    // Every copy maps back to the one engine node, for gradient norms.
    expect(convs.slice(1).every((layer) => built.layerToNode.get(layer.name) === "conv2d_2")).toBe(true);
    built.model.dispose();
  });

  test("the same seed gives the same weights, and shared layers start alike across variants", () => {
    const first = (id: string, seed: number) => {
      const m = toTf(catalogModel(id)!, { seed });
      const values = Array.from(m.getLayer("conv2d").getWeights()[0].dataSync().slice(0, 8));
      m.dispose();
      return values;
    };
    expect(first("cifar-cnn", 3)).toEqual(first("cifar-cnn", 3));
    expect(first("cifar-cnn", 3)).toEqual(first("cifar-cnn-skip", 3));
    expect(first("cifar-cnn", 3)).not.toEqual(first("cifar-cnn", 4));
  });

  test("max pooling pads with a very negative number, like PyTorch", () => {
    const b = new Builder("Pool", "vision");
    const x = b.input("1x2x2");
    b.node("maxpool", x, { k: 3, s: 1, p: 1 });
    const m = toTf(b.build());
    const out = m.predict(tf.tensor4d([-1, -2, -3, -4], [1, 2, 2, 1])) as tf.Tensor;
    expect(Array.from(out.dataSync())).toEqual([-1, -1, -1, -1]);
    m.dispose();
  });
});
