// The one graph format every architecture is turned into: a classic CNN from
// the catalog, an LLM read from its config.json, an ONNX file or a design
// written by hand. Everything else (shapes, counts, diffs, drawing, code
// export, training) works on this.

/** A tensor's shape without the batch dimension: [C, H, W] for images, [seq, features] for sequences. */
export type Shape = number[];

export type ArgValue = number | string | boolean;

export interface Node {
  id: string;
  /** The operation, a key of the op registry. */
  op: string;
  args: Record<string, ArgValue>;
  /** Ids of the nodes this one reads from, in order. The model input is the node with op "input". */
  inputs: string[];
  /** A readable name; defaults to the op. */
  label?: string;
  /** The block this node belongs to. */
  group?: string;
  /** Where it came from in the text format, for error messages. */
  line?: number;
}

export interface Group {
  id: string;
  label: string;
  /** What sort of block this is, such as "bottleneck", "mbconv" or "decoder". Used to line blocks up in a diff. */
  kind: string;
  parent?: string;
  /**
   * How many identical copies of this block the model stacks. The nodes are
   * listed once; counts are multiplied. The block's output shape must equal
   * its input shape for this to be valid (true for transformer layers).
   */
  repeat?: number;
}

export type ModelKind = "vision" | "language" | "tabular";

export interface Model {
  name: string;
  /** Where it came from: "catalog", "huggingface:<repo>", "onnx", "text". */
  source: string;
  kind: ModelKind;
  nodes: Node[];
  groups: Group[];
  /**
   * Headline settings, compared field by field in a diff: hidden size,
   * layers, heads, attention type, experts and so on.
   */
  meta: Record<string, ArgValue>;
  /** Sequence length used for shapes and compute on language models. */
  seqLen?: number;
}

/** What the engine works out for one node. Counts are for one copy, before a block's repeat. */
export interface NodeInfo {
  shape: Shape;
  params: number;
  /** Parameters used for one token or one image; less than `params` for mixture-of-experts layers. */
  activeParams: number;
  /** Floating point operations for one forward pass of one example. */
  flops: number;
  /** How many times the node runs: the product of the repeats of its blocks. */
  times: number;
  error?: string;
}

export interface Analysis {
  info: Map<string, NodeInfo>;
  params: number;
  activeParams: number;
  flops: number;
  outputShape: Shape | null;
  errors: { node: string; message: string; line?: number }[];
}
