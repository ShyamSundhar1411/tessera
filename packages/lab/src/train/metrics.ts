// Plain number crunching for evaluating classifiers and comparing training
// runs. Kept free of tf.js so it is easy to test and reuse in the UI.

export interface ClassScores {
  precision: number;
  recall: number;
  f1: number;
  /** How many test examples truly belong to this class. */
  support: number;
}

export interface Evaluation {
  loss: number;
  accuracy: number;
  errorRate: number;
  /** Only when there are more than 5 classes, otherwise it would always be 1. */
  top5Accuracy?: number;
  perClass: ClassScores[];
  macro: { precision: number; recall: number; f1: number };
  /** Rows are the true class, columns the predicted class. */
  confusion: number[][];
  examples: number;
  /** exp(mean loss per token), for tasks scored per token (language modelling, masked tokens, seq2seq). */
  perplexity?: number;
  /** seq2seq only: share of test sequences produced entirely right. */
  sequenceAccuracy?: number;
}

/** Counts how often each true class (row) was predicted as each class (column). */
export function confusionMatrix(truth: ArrayLike<number>, predicted: ArrayLike<number>, classes: number): number[][] {
  if (truth.length !== predicted.length) throw new Error(`confusionMatrix needs as many predictions as labels, got ${predicted.length} and ${truth.length}`);
  const matrix = Array.from({ length: classes }, () => new Array<number>(classes).fill(0));
  for (let i = 0; i < truth.length; i++) matrix[truth[i]][predicted[i]]++;
  return matrix;
}

/**
 * Precision, recall and F1 for every class from a confusion matrix. A class
 * that is never predicted (or never present) gets 0 rather than NaN, which is
 * what scikit-learn reports with its default zero_division warning.
 */
export function classScores(confusion: number[][]): ClassScores[] {
  const n = confusion.length;
  return confusion.map((row, c) => {
    const truePositive = row[c];
    const support = row.reduce((a, b) => a + b, 0);
    let predictedAs = 0;
    for (let r = 0; r < n; r++) predictedAs += confusion[r][c];
    const precision = predictedAs ? truePositive / predictedAs : 0;
    const recall = support ? truePositive / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}

/** The unweighted average over classes, so rare classes count as much as common ones. */
export function macroAverage(scores: ClassScores[]): Evaluation["macro"] {
  const n = scores.length || 1;
  const sum = (key: "precision" | "recall" | "f1") => scores.reduce((a, s) => a + s[key], 0) / n;
  return { precision: sum("precision"), recall: sum("recall"), f1: sum("f1") };
}

/** Builds a full evaluation from labels, predictions and a loss worked out elsewhere. */
export function evaluate(args: { truth: ArrayLike<number>; predicted: ArrayLike<number>; top5Hits?: number; loss: number; classes: number }): Evaluation {
  const { truth, predicted, loss, classes } = args;
  const confusion = confusionMatrix(truth, predicted, classes);
  const examples = truth.length;
  let correct = 0;
  for (let c = 0; c < classes; c++) correct += confusion[c][c];
  const accuracy = examples ? correct / examples : 0;
  const perClass = classScores(confusion);
  return {
    loss,
    accuracy,
    errorRate: 1 - accuracy,
    top5Accuracy: classes > 5 && args.top5Hits !== undefined && examples ? args.top5Hits / examples : undefined,
    perClass,
    macro: macroAverage(perClass),
    confusion,
    examples,
  };
}

export interface Stats {
  mean: number;
  /** Sample standard deviation (divides by n - 1); 0 for a single run. */
  std: number;
  min: number;
  max: number;
  n: number;
}

export function stats(values: number[]): Stats {
  const n = values.length;
  if (!n) return { mean: NaN, std: NaN, min: NaN, max: NaN, n: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = n > 1 ? values.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1) : 0;
  return { mean, std: Math.sqrt(variance), min: Math.min(...values), max: Math.max(...values), n };
}

/** The final metrics summarised across seeds. */
export type RunMetric = "loss" | "accuracy" | "errorRate" | "macroF1";
export const RUN_METRICS: RunMetric[] = ["loss", "accuracy", "errorRate", "macroF1"];
const HIGHER_IS_BETTER: Record<CompareMetric, boolean> = { loss: false, accuracy: true, errorRate: false, macroF1: true, perplexity: false };

export const metricOf = (evaluation: Evaluation, metric: RunMetric): number =>
  metric === "macroF1" ? evaluation.macro.f1 : evaluation[metric];

/** Perplexity joins the run metrics for tasks scored per token. */
export type CompareMetric = RunMetric | "perplexity";

export interface MetricComparison {
  metric: CompareMetric;
  meanA: number;
  meanB: number;
  /** meanB - meanA. */
  difference: number;
  /** Which side is better on average, or "tie" when the means are equal. */
  better: "a" | "b" | "tie";
  /**
   * True when the gap between the means is larger than stdA + stdB, a rough
   * "beyond seed noise" check, not a significance test. Null when either side
   * has fewer than two runs, because one run says nothing about the spread.
   */
  beyondNoise: boolean | null;
  test: "mean gap > stdA + stdB";
}

/** A summary of final metrics across seeds; perplexity is present only for per-token tasks. */
export type RunSummary = Record<RunMetric, Stats> & { perplexity?: Stats };

/**
 * Compares two models' final metrics across seeds. Takes anything with a
 * `summary` of per-metric stats, such as a model's training result.
 * Perplexity is compared too when both summaries have it.
 */
export function compareRuns(a: { summary: RunSummary }, b: { summary: RunSummary }): MetricComparison[] {
  const metrics: CompareMetric[] = a.summary.perplexity && b.summary.perplexity ? [...RUN_METRICS, "perplexity"] : RUN_METRICS;
  return metrics.map((metric) => {
    const sa = a.summary[metric]!;
    const sb = b.summary[metric]!;
    const difference = sb.mean - sa.mean;
    const better = difference === 0 ? "tie" : (difference > 0) === HIGHER_IS_BETTER[metric] ? "b" : "a";
    const beyondNoise = sa.n < 2 || sb.n < 2 ? null : Math.abs(difference) > sa.std + sb.std;
    return { metric, meanA: sa.mean, meanB: sb.mean, difference, better, beyondNoise, test: "mean gap > stdA + stdB" };
  });
}

/** Mean and standard deviation per epoch across runs, for drawing a band. Runs cut short are averaged over the epochs they reached. */
export function band(curves: number[][]): { mean: number[]; std: number[] } {
  const epochs = Math.max(0, ...curves.map((c) => c.length));
  const mean: number[] = [];
  const std: number[] = [];
  for (let e = 0; e < epochs; e++) {
    const s = stats(curves.filter((c) => e < c.length).map((c) => c[e]));
    mean.push(s.mean);
    std.push(s.std);
  }
  return { mean, std };
}
