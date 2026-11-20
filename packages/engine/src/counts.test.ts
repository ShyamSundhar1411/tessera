import { describe, expect, test } from "bun:test";
import { analyze } from "./analyze";
import { catalogModel } from "./catalog";

// Every catalog model must match the parameter count its authors publish. The
// image models match torchvision to the parameter; language models match the
// count on their model card to within half a percent.
const exact: [string, number][] = [
  ["vgg16", 138_357_544],
  ["resnet18", 11_689_512],
  ["resnet50", 25_557_032],
  ["mobilenet-v2", 3_504_872],
  ["efficientnet-b0", 5_288_548],
  ["gpt2", 124_439_808],
  ["smollm2-135m", 134_515_008],
];

const close: [string, number, number?][] = [
  ["llama-3.1-8b", 8.03e9],
  ["mistral-7b", 7.24e9],
  ["mixtral-8x7b", 46.7e9, 12.9e9],
  ["qwen2.5-7b", 7.62e9],
  ["qwen3-8b", 8.19e9],
  ["qwen3-30b-a3b", 30.5e9, 3.3e9],
  ["deepseek-v3", 671e9, 37e9],
  ["gemma-2-9b", 9.24e9],
  ["gemma-3-1b", 1.0e9],
  ["phi-3-mini", 3.82e9],
];

describe("parameter counts match the published numbers", () => {
  for (const [id, expected] of exact) {
    test(id, () => {
      const analysis = analyze(catalogModel(id)!);
      expect(analysis.errors).toEqual([]);
      expect(analysis.params).toBe(expected);
    });
  }
  for (const [id, total, active] of close) {
    test(id, () => {
      const analysis = analyze(catalogModel(id)!);
      expect(analysis.errors).toEqual([]);
      expect(Math.abs(analysis.params - total) / total).toBeLessThan(0.005);
      if (active) expect(Math.abs(analysis.activeParams - active) / active).toBeLessThan(0.03);
    });
  }
});

describe("shapes", () => {
  test("ResNet-50 ends in 1000 classes", () => {
    expect(analyze(catalogModel("resnet50")!).outputShape).toEqual([1000]);
  });
  test("a language model ends in vocabulary scores for every position", () => {
    expect(analyze(catalogModel("gpt2")!).outputShape).toEqual([2048, 50257]);
  });
});
