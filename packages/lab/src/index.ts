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
  type CompareMetric,
  type RunSummary,
  type MetricComparison,
} from "./train/metrics";
export { buildGraph, GraphError, disposeTrace, traceArrays, type GraphModel, type GraphOptions, type ApplyOptions, type TraceStep, type TracedLayer, type PlainFeed } from "./train/graph";
export {
  prepareForTraining,
  planTraining,
  loadTaskData,
  loadShakespeare,
  encodeChars,
  decodeChars,
  CHAR_VOCAB,
  CHAR_MASK,
  SYMBOLS,
  SHAKESPEARE_URL,
  type TaskKind,
  type TaskSpec,
  type InputRole,
  type PreparedModel,
  type PrepareResult,
  type PrepareOptions,
  type TaskData,
  type TokenSplit,
} from "./train/tasks";
export { trainCompareAny, planCompare, TaskMismatchError, type AnyTrainOptions, type AnyTrainEvent, type AnyModelResult, type AnyTrainResult } from "./train/trainAny";
export { PRETRAINED, IMAGENET_LABELS, labelsFor, pretrainedEntry, type PretrainedEntry, type InputSpec } from "./run/registry";
export { classify, loadSession, releaseSession, type ClassifyResult, type ClassifyProgress, type Prediction, type ClassifiableImage } from "./run/classify";
export { cropRect, toNchw, softmax, topK } from "./run/preprocess";
export { loadGpt2, nextTokenLogits, gpt2Model, gpt2Assignments, assignWeights, parseSafetensors, GPT2_FILES, GPT2_BYTES, type Gpt2, type Gpt2Progress, type SafeTensor } from "./run/gpt2";
export { Gpt2Tokenizer } from "./run/bpe";
