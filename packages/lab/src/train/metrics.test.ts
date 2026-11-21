import { describe, expect, test } from "bun:test";
import { band, classScores, compareRuns, confusionMatrix, evaluate, macroAverage, stats, type RunMetric, type Stats } from "./metrics";

describe("classification metrics", () => {
  // Three classes, eight examples, worked out by hand.
  const truth = [0, 0, 0, 1, 1, 2, 2, 2];
  const predicted = [0, 0, 1, 1, 2, 2, 2, 0];

  test("confusion matrix puts true classes on rows", () => {
    expect(confusionMatrix(truth, predicted, 3)).toEqual([
      [2, 1, 0],
      [0, 1, 1],
      [1, 0, 2],
    ]);
  });

  test("precision, recall and F1 per class", () => {
    const scores = classScores(confusionMatrix(truth, predicted, 3));
    // Class 0: predicted 3 times, 2 right; 3 present, 2 found.
    expect(scores[0].precision).toBeCloseTo(2 / 3);
    expect(scores[0].recall).toBeCloseTo(2 / 3);
    expect(scores[0].f1).toBeCloseTo(2 / 3);
    // Class 1: predicted twice, 1 right; 2 present, 1 found.
    expect(scores[1].precision).toBeCloseTo(0.5);
    expect(scores[1].recall).toBeCloseTo(0.5);
    // Class 2: predicted 3 times, 2 right; 3 present, 2 found.
    expect(scores[2].f1).toBeCloseTo(2 / 3);
    expect(scores.map((s) => s.support)).toEqual([3, 2, 3]);
    const macro = macroAverage(scores);
    expect(macro.f1).toBeCloseTo((2 / 3 + 0.5 + 2 / 3) / 3);
  });

  test("a class never predicted scores 0, not NaN", () => {
    const scores = classScores(confusionMatrix([0, 1], [0, 0], 2));
    expect(scores[1]).toEqual({ precision: 0, recall: 0, f1: 0, support: 1 });
    expect(scores[0].precision).toBe(0.5);
    expect(scores[0].recall).toBe(1);
    expect(scores[0].f1).toBeCloseTo(2 / 3);
  });

  test("evaluate fills in accuracy, error rate and top-5 only when there are more than 5 classes", () => {
    const small = evaluate({ truth, predicted, top5Hits: 8, loss: 0.7, classes: 3 });
    expect(small.accuracy).toBe(5 / 8);
    expect(small.errorRate).toBe(3 / 8);
    expect(small.top5Accuracy).toBeUndefined();
    const big = evaluate({ truth, predicted, top5Hits: 7, loss: 0.7, classes: 10 });
    expect(big.top5Accuracy).toBe(7 / 8);
    expect(big.confusion.length).toBe(10);
  });
});

describe("run statistics", () => {
  test("mean, sample standard deviation, min and max", () => {
    const s = stats([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(s.mean).toBe(5);
    expect(s.std).toBeCloseTo(Math.sqrt(32 / 7));
    expect(s.min).toBe(2);
    expect(s.max).toBe(9);
    expect(stats([3]).std).toBe(0);
  });

  test("band averages each epoch across runs", () => {
    const { mean, std } = band([
      [1, 2, 3],
      [3, 4],
    ]);
    expect(mean).toEqual([2, 3, 3]);
    expect(std[0]).toBeCloseTo(Math.SQRT2);
    expect(std[2]).toBe(0);
  });

  const summary = (values: Partial<Record<RunMetric, number[]>>) =>
    ({ summary: Object.fromEntries((["loss", "accuracy", "errorRate", "macroF1"] as RunMetric[]).map((m) => [m, stats(values[m] ?? [0.5, 0.5])])) as Record<RunMetric, Stats> });

  test("compareRuns says beyond noise only when the gap beats the combined spread", () => {
    const a = summary({ accuracy: [0.8, 0.82, 0.81], loss: [0.6, 0.5, 0.7] });
    const b = summary({ accuracy: [0.9, 0.91, 0.89], loss: [0.55, 0.65, 0.6] });
    const result = Object.fromEntries(compareRuns(a, b).map((c) => [c.metric, c]));
    expect(result.accuracy.better).toBe("b");
    expect(result.accuracy.difference).toBeCloseTo(0.09);
    expect(result.accuracy.beyondNoise).toBe(true);
    expect(result.loss.better).toBe("a");
    expect(result.loss.beyondNoise).toBe(false);
  });

  test("one seed cannot say anything about noise", () => {
    const [accuracy] = compareRuns(summary({ loss: [0.4] }), summary({ loss: [0.3] }));
    expect(accuracy.metric).toBe("loss");
    expect(accuracy.better).toBe("b");
    expect(accuracy.beyondNoise).toBeNull();
  });
});
