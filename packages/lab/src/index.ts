// Training in the browser with tf.js, and running pretrained models with ONNX Runtime Web.
export { toTf, buildTf, trainingProblem, trainableParams, UnsupportedOpError, type TfBuild, type TfOptions } from "./train/toTf";
export { loadDataset, makeDataset, disposeDataset, DATASETS, type Dataset, type DatasetName, type DatasetInfo, type DatasetProgress } from "./train/datasets";
export {
  trainCompare,
  checkCompatibility,
  initBackend,
  evaluateModel,
  summarise,
  DatasetMismatchError,
  type TrainOptions,
  type TrainEvent,
  type TrainResult,
  type ModelResult,
  type RunResult,
  type EpochRecord,
  type Curve,
} from "./train/train";
export {
  compareRuns,
  confusionMatrix,
  classScores,
  macroAverage,
  evaluate,
  stats,
  band,
  RUN_METRICS,
  type Evaluation,
  type ClassScores,
  type Stats,
  type RunMetric,
  type MetricComparison,
} from "./train/metrics";
export { PRETRAINED, IMAGENET_LABELS, labelsFor, pretrainedEntry, type PretrainedEntry, type InputSpec } from "./run/registry";
export { classify, loadSession, releaseSession, type ClassifyResult, type ClassifyProgress, type Prediction, type ClassifiableImage } from "./run/classify";
export { cropRect, toNchw, softmax, topK } from "./run/preprocess";
