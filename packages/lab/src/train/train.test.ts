import { describe, expect, test } from "bun:test";
import { catalogModel } from "tessera-engine";
import { DATASETS, makeDataset } from "./datasets";
import { compareRuns } from "./metrics";
import { checkCompatibility, trainCompare, type TrainEvent } from "./train";

// Random images stand in for MNIST: the point is that the loop runs end to end.
function randomMnist(train: number, test: number) {
  let s = 1;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const labels = (n: number) => Uint8Array.from({ length: n }, () => Math.floor(rand() * 10));
  const pixels = (n: number) => Float32Array.from({ length: n * 784 }, rand);
  const info = DATASETS.mnist;
  return makeDataset({ name: "mnist", label: info.label, shape: info.shape, classes: info.classes, trainPixels: pixels(train), trainLabels: labels(train), testPixels: pixels(test), testLabels: labels(test) });
}

describe("trainCompare", () => {
  test("explains an input shape mismatch", () => {
    expect(checkCompatibility(catalogModel("cifar-cnn")!, DATASETS.mnist)).toBe("Small CNN expects 3×32×32 images, MNIST is 1×28×28");
    expect(checkCompatibility(catalogModel("lenet5")!, DATASETS.mnist)).toBeNull();
  });

  test("trains two models over two seeds and summarises them", async () => {
    const data = randomMnist(64, 40);
    const events: TrainEvent[] = [];
    const result = await trainCompare(
      [catalogModel("mnist-mlp")!, catalogModel("lenet5")!],
      data,
      { epochs: 1, batchSize: 32, learningRate: 0.001, seed: 1, seeds: [1, 2], backends: ["cpu"] },
      (event) => events.push(event),
    );
    expect(result.backend).toBe("cpu");
    expect(result.aborted).toBe(false);
    // 2 batches per epoch, so 2 batch events and 1 epoch event per run, 4 runs.
    expect(events.filter((e) => e.type === "batch").length).toBe(8);
    const epochs = events.filter((e) => e.type === "epoch");
    expect(epochs.map((e) => [e.modelIndex, e.seed])).toEqual([[0, 1], [1, 1], [0, 2], [1, 2]]);
    const first = epochs[0];
    expect(Number.isFinite(first.loss)).toBe(true);
    expect(first.evaluation!.confusion.flat().reduce((a, b) => a + b, 0)).toBe(40);
    expect(first.evaluation!.top5Accuracy).toBeGreaterThanOrEqual(0);
    expect(Object.keys(first.gradientNorms!)).toContain("linear_3");
    for (const model of result.models) {
      expect(model.runs.length).toBe(2);
      expect(model.summary.accuracy.n).toBe(2);
      expect(model.curves.valAccuracy.mean.length).toBe(1);
    }
    expect(result.models[0].params).toBe(109_386);
    expect(compareRuns(result.models[0], result.models[1]).length).toBe(4);
  }, 60_000);

  test("stops when aborted", async () => {
    const data = randomMnist(96, 16);
    const controller = new AbortController();
    const result = await trainCompare(
      [catalogModel("mnist-mlp")!],
      data,
      { epochs: 5, batchSize: 16, learningRate: 0.001, seed: 1, backends: ["cpu"], reportsPerEpoch: 6 },
      (event) => {
        if (event.type === "batch" && event.batch === 2) controller.abort();
      },
      controller.signal,
    );
    expect(result.aborted).toBe(true);
    expect(result.models[0].runs[0].final).toBeNull();
  });

  test("the same seed gives the same run", async () => {
    const data = randomMnist(32, 16);
    const run = async () => (await trainCompare([catalogModel("mnist-mlp")!], data, { epochs: 1, batchSize: 16, learningRate: 0.01, seed: 7, backends: ["cpu"] }, () => {})).models[0].summary.loss.mean;
    expect(await run()).toBe(await run());
  });
});
