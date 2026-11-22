import { describe, expect, test } from "bun:test";
import { catalogModel } from "tessera-engine";
import { DATASETS, makeDataset } from "./datasets";
import { compareRuns } from "./metrics";
import { planTraining, prepareForTraining } from "./tasks";
import { TaskMismatchError, trainCompareAny, type AnyTrainEvent, type AnyTrainOptions } from "./trainAny";

// A short fixed text stands in for tiny Shakespeare, so the tests need no network.
const TEXT = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n".repeat(6);
const tiny = { dim: 32, layers: 2, heads: 2, seqLen: 16, seq2seqLength: 6 };
const base: AnyTrainOptions = { epochs: 3, batchSize: 8, learningRate: 0.005, seed: 1, backends: ["cpu"], text: TEXT, trainExamples: 80, testExamples: 16, prepare: tiny, reportsPerEpoch: 2 };

function randomCifar(train: number, test: number) {
  let s = 7;
  const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const labels = (n: number) => Uint8Array.from({ length: n }, () => Math.floor(rand() * 10));
  const pixels = (n: number) => Float32Array.from({ length: n * 3 * 32 * 32 }, rand);
  const info = DATASETS.cifar10;
  return makeDataset({ name: "cifar10", label: info.label, shape: info.shape, classes: info.classes, trainPixels: pixels(train), trainLabels: labels(train), testPixels: pixels(test), testLabels: labels(test) });
}

/** Trains one model for 3 epochs of 10 steps and returns the per-epoch training losses. */
async function epochLosses(id: string, options: Partial<AnyTrainOptions> = {}) {
  const events: AnyTrainEvent[] = [];
  const started = performance.now();
  const result = await trainCompareAny([catalogModel(id)!], { ...base, ...options }, (e) => events.push(e));
  const seconds = (performance.now() - started) / 1000;
  console.log(`${id}: ${result.models[0].description} ${seconds.toFixed(1)}s; epoch losses ${events.filter((e) => e.type === "epoch").map((e) => e.loss.toFixed(3)).join(", ")}`);
  return { losses: events.filter((e) => e.type === "epoch").map((e) => e.loss), events, result };
}

describe("task picking", () => {
  test("decoders, encoders, encoder-decoders and images each get their task", () => {
    const kind = (id: string) => {
      const p = prepareForTraining(catalogModel(id)!);
      return p.ok ? p.task.kind : p.reason;
    };
    expect(kind("gpt2")).toBe("language-modelling");
    expect(kind("deepseek-v3")).toBe("language-modelling");
    expect(kind("bert-base")).toBe("masked-tokens");
    expect(kind("distilbert-base")).toBe("masked-tokens");
    expect(kind("transformer-base")).toBe("seq2seq");
    expect(kind("vit-b16")).toBe("image-classification");
    expect(kind("cifar-cnn")).toBe("image-classification");
    const cls = prepareForTraining(catalogModel("bert-base")!, { encoderTask: "sequence-classification" });
    expect(cls.ok && cls.task.kind).toBe("sequence-classification");
  });

  test("shrunk sizes are reported", () => {
    const p = prepareForTraining(catalogModel("llama-3.1-8b")!);
    if (!p.ok) throw new Error(p.reason);
    expect(p.originalParams).toBeGreaterThan(8e9);
    expect(p.params).toBeLessThan(1e6);
    expect(p.description).toContain("shrunk to hidden 64");
  });

  test("models with different tasks are refused with a reason", async () => {
    const plan = planTraining([catalogModel("gpt2")!, catalogModel("bert-base")!]);
    expect(plan.ok).toBe(false);
    if (!plan.ok) expect(plan.reason).toContain("learn different things");
    expect(trainCompareAny([catalogModel("gpt2")!, catalogModel("cifar-cnn")!], base, () => {})).rejects.toBeInstanceOf(TaskMismatchError);
  });
});

describe("loss goes down in about 30 steps", () => {
  test("tiny decoder: next character of a fixed text", async () => {
    const { losses, result } = await epochLosses("llama-3.1-8b");
    expect(losses[2]).toBeLessThan(losses[0] * 0.9);
    const final = result.models[0].runs[0].final!;
    expect(final.perplexity).toBeCloseTo(Math.exp(final.loss), 5);
    expect(result.models[0].summary.perplexity!.n).toBe(1);
  }, 60_000);

  test("tiny encoder-decoder: reverse a sequence", async () => {
    const { losses, result } = await epochLosses("transformer-base");
    expect(result.task.kind).toBe("seq2seq");
    expect(losses[2]).toBeLessThan(losses[0] * 0.95);
    expect(result.models[0].runs[0].final!.sequenceAccuracy).toBeGreaterThanOrEqual(0);
  }, 60_000);

  test("tiny vision transformer on random images", async () => {
    const { losses } = await epochLosses("vit-b16", { imageDataset: randomCifar(80, 16) });
    expect(losses[2]).toBeLessThan(losses[0]);
  }, 60_000);

  test("encoder: masked characters, and mixture of experts with latent attention", async () => {
    const bert = await epochLosses("bert-base");
    expect(bert.losses[2]).toBeLessThan(bert.losses[0]);
    const deepseek = await epochLosses("deepseek-v3", { epochs: 2 });
    expect(deepseek.losses[1]).toBeLessThan(deepseek.losses[0]);
  }, 60_000);
});

describe("trainCompareAny", () => {
  test("two decoders over two seeds: events, summaries and the comparison with perplexity", async () => {
    const events: AnyTrainEvent[] = [];
    const result = await trainCompareAny([catalogModel("gpt2")!, catalogModel("qwen3-8b")!], { ...base, epochs: 1, seeds: [1, 2] }, (e) => events.push(e));
    expect(result.backend).toBe("cpu");
    expect(result.task.kind).toBe("language-modelling");
    const epochs = events.filter((e) => e.type === "epoch");
    expect(epochs.map((e) => [e.modelIndex, e.seed])).toEqual([[0, 1], [1, 1], [0, 2], [1, 2]]);
    expect(epochs[0].valPerplexity).toBeGreaterThan(1);
    expect(Object.keys(epochs[0].gradientNorms!).length).toBeGreaterThan(5);
    const comparison = compareRuns(result.models[0], result.models[1]);
    expect(comparison.map((c) => c.metric)).toEqual(["loss", "accuracy", "errorRate", "macroF1", "perplexity"]);
    expect(comparison.every((c) => c.beyondNoise !== null)).toBe(true);
    expect(result.models[0].curves.valPerplexity!.mean.length).toBe(1);
  }, 60_000);

  test("the same seed gives the same run", async () => {
    const run = async () => (await trainCompareAny([catalogModel("smollm2-135m")!], { ...base, epochs: 1 }, () => {})).models[0].summary.loss.mean;
    expect(await run()).toBe(await run());
  }, 60_000);

  test("stops when aborted", async () => {
    const controller = new AbortController();
    const result = await trainCompareAny(
      [catalogModel("gpt2")!],
      { ...base, reportsPerEpoch: 10 },
      (event) => {
        if (event.type === "batch" && event.batch === 2) controller.abort();
      },
      controller.signal,
    );
    expect(result.aborted).toBe(true);
    expect(result.models[0].runs[0].final).toBeNull();
  }, 60_000);

  test("sequence classification on a generated task", async () => {
    const result = await trainCompareAny([catalogModel("bert-base")!], { ...base, epochs: 1, prepare: { ...tiny, encoderTask: "sequence-classification" } }, () => {});
    expect(result.task.kind).toBe("sequence-classification");
    expect(result.models[0].runs[0].final!.confusion.length).toBe(4);
    expect(result.models[0].summary.perplexity).toBeUndefined();
  }, 60_000);
});
