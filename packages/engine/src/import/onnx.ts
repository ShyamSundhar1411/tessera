import { Builder } from "../builder";
import type { ArgValue, Group, Model } from "../types";
import { fields, float32, pushFloats, pushInts, text } from "./protobuf";

// Turns an ONNX file into the engine's graph. ONNX stores a model as a flat,
// topologically sorted list of operator nodes wired together by tensor names,
// with the weights as named "initializer" tensors. The engine wants layers, so
// weights are read only for their shapes (a convolution's output channels, a
// dense layer's width) and the operators the engine models are mapped onto
// its ops. Field numbers follow onnx/onnx.proto.

export class OnnxError extends Error {}

interface Tensor {
  name: string;
  dims: number[];
  /** TensorProto.DataType: 1 float, 6 int32, 7 int64, 11 double. */
  dataType: number;
  /** A view into the file, never a copy: weights can be hundreds of megabytes. */
  raw?: Uint8Array;
  floats: number[];
  ints: number[];
}

interface Attribute {
  f?: number;
  i?: number;
  s?: string;
  t?: Tensor;
  ints: number[];
  floats: number[];
}

interface OnnxNode {
  name: string;
  op: string;
  inputs: string[];
  outputs: string[];
  attrs: Map<string, Attribute>;
}

interface ValueInfo {
  name: string;
  /** Dimension sizes; null where the file names the dimension instead of fixing it (such as "batch"). */
  dims: (number | null)[];
}

// -- Decoding the protobuf messages -------------------------------------------

function readTensor(buf: Uint8Array): Tensor {
  const tensor: Tensor = { name: "", dims: [], dataType: 0, floats: [], ints: [] };
  for (const f of fields(buf)) {
    if (f.field === 1) pushInts(tensor.dims, f);
    else if (f.field === 2) tensor.dataType = f.int;
    else if (f.field === 4) pushFloats(tensor.floats, f);
    else if (f.field === 5 || f.field === 7) pushInts(tensor.ints, f);
    else if (f.field === 8) tensor.name = text(f.bytes);
    else if (f.field === 9) tensor.raw = f.bytes;
  }
  return tensor;
}

function readAttribute(buf: Uint8Array): [string, Attribute] {
  let name = "";
  const attr: Attribute = { ints: [], floats: [] };
  for (const f of fields(buf)) {
    if (f.field === 1) name = text(f.bytes);
    else if (f.field === 2) attr.f = float32(f.bytes);
    else if (f.field === 3) attr.i = f.int;
    else if (f.field === 4) attr.s = text(f.bytes);
    else if (f.field === 5) attr.t = readTensor(f.bytes);
    else if (f.field === 7) pushFloats(attr.floats, f);
    else if (f.field === 8) pushInts(attr.ints, f);
  }
  return [name, attr];
}

function readNode(buf: Uint8Array): OnnxNode {
  const node: OnnxNode = { name: "", op: "", inputs: [], outputs: [], attrs: new Map() };
  for (const f of fields(buf)) {
    if (f.field === 1) node.inputs.push(text(f.bytes));
    else if (f.field === 2) node.outputs.push(text(f.bytes));
    else if (f.field === 3) node.name = text(f.bytes);
    else if (f.field === 4) node.op = text(f.bytes);
    else if (f.field === 5) {
      const [name, attr] = readAttribute(f.bytes);
      node.attrs.set(name, attr);
    }
  }
  return node;
}

/** ValueInfoProto: name 1, type 2 -> TypeProto.tensor_type 1 -> shape 2 -> dim 1 -> dim_value 1. */
function readValueInfo(buf: Uint8Array): ValueInfo {
  const info: ValueInfo = { name: "", dims: [] };
  const child = (msg: Uint8Array, field: number): Uint8Array | undefined => {
    let found: Uint8Array | undefined;
    for (const f of fields(msg)) if (f.field === field) found = f.bytes;
    return found;
  };
  let type: Uint8Array | undefined;
  for (const f of fields(buf)) {
    if (f.field === 1) info.name = text(f.bytes);
    else if (f.field === 2) type = f.bytes;
  }
  const tensorType = type && child(type, 1);
  const shape = tensorType && child(tensorType, 2);
  if (shape) {
    for (const f of fields(shape)) {
      if (f.field !== 1) continue;
      let size: number | null = null;
      for (const d of fields(f.bytes)) if (d.field === 1) size = d.int;
      info.dims.push(size !== null && size > 0 ? size : null);
    }
  }
  return info;
}

/** The first few values of a small tensor, such as a Clip bound or a Reshape target. */
function values(tensor: Tensor, limit = 16): number[] {
  if (tensor.floats.length) return tensor.floats.slice(0, limit);
  if (tensor.ints.length) return tensor.ints.slice(0, limit);
  const raw = tensor.raw;
  if (!raw) return [];
  const view = new DataView(raw.buffer, raw.byteOffset, raw.length);
  const out: number[] = [];
  const width = tensor.dataType === 7 || tensor.dataType === 11 ? 8 : 4;
  for (let pos = 0; pos + width <= raw.length && out.length < limit; pos += width) {
    if (tensor.dataType === 1) out.push(view.getFloat32(pos, true));
    else if (tensor.dataType === 6) out.push(view.getInt32(pos, true));
    else if (tensor.dataType === 7) out.push(Number(view.getBigInt64(pos, true)));
    else if (tensor.dataType === 11) out.push(view.getFloat64(pos, true));
    else break;
  }
  return out;
}

/** A Constant node's value, from whichever of its attributes is set. */
function constantValue(node: OnnxNode): Tensor {
  const value = node.attrs.get("value");
  if (value?.t) return { ...value.t, name: node.outputs[0] ?? value.t.name };
  const tensor: Tensor = { name: node.outputs[0] ?? "", dims: [], dataType: 0, floats: [], ints: [] };
  const float = node.attrs.get("value_float");
  const int = node.attrs.get("value_int");
  const floats = node.attrs.get("value_floats");
  const ints = node.attrs.get("value_ints");
  if (float?.f !== undefined) tensor.floats = [float.f];
  else if (int?.i !== undefined) tensor.ints = [int.i];
  else if (floats) {
    tensor.floats = floats.floats;
    tensor.dims = [floats.floats.length];
  } else if (ints) {
    tensor.ints = ints.ints;
    tensor.dims = [ints.ints.length];
  }
  return tensor;
}

// -- Groups from PyTorch's scoped names ---------------------------------------

/** "blocks.1.0" -> "blocks", "layer1" -> "layer": the module type behind a numbered module. */
const kindOf = (segment: string) => segment.replace(/[\d.]+/g, "").replace(/_+$/, "") || "block";

/**
 * PyTorch names exported nodes after the module path that made them, such as
 * "/layer1/layer1.0/conv1/Conv". Each path prefix becomes a nested group, so a
 * ResNet falls into its stages and blocks. Groups that would hold a single
 * node (the "conv1" module around one Conv) are dropped as clutter, and their
 * node joins the nearest group above.
 */
function buildGroups(paths: Map<string, string[]>): { groups: Group[]; groupOf: Map<string, string> } {
  interface Scope {
    key: string;
    segments: string[];
    nodes: string[];
    total: number;
  }
  const scopes = new Map<string, Scope>();
  for (const [nodeId, segments] of paths) {
    for (let depth = 1; depth <= segments.length; depth++) {
      const key = segments.slice(0, depth).join("/");
      let scope = scopes.get(key);
      if (!scope) scopes.set(key, (scope = { key, segments: segments.slice(0, depth), nodes: [], total: 0 }));
      scope.total++;
      if (depth === segments.length) scope.nodes.push(nodeId);
    }
  }

  const kept = new Map<string, string>();
  const groups: Group[] = [];
  const counts = new Map<string, number>();
  // Map insertion order puts every parent before its children.
  for (const scope of scopes.values()) {
    if (scope.total < 2) continue;
    const parentSegments = scope.segments.slice(0, -1);
    let parent: string | undefined;
    for (let depth = parentSegments.length; depth > 0 && !parent; depth--) parent = kept.get(parentSegments.slice(0, depth).join("/"));
    const label = scope.segments[scope.segments.length - 1];
    const kind = kindOf(parentSegments.length ? parentSegments[parentSegments.length - 1] : label);
    const n = (counts.get(kind) ?? 0) + 1;
    counts.set(kind, n);
    const id = n === 1 ? `g_${kind}` : `g_${kind}_${n}`;
    kept.set(scope.key, id);
    groups.push({ id, label, kind, parent });
  }

  const groupOf = new Map<string, string>();
  for (const [nodeId, segments] of paths) {
    for (let depth = segments.length; depth > 0; depth--) {
      const id = kept.get(segments.slice(0, depth).join("/"));
      if (id) {
        groupOf.set(nodeId, id);
        break;
      }
    }
  }
  return { groups, groupOf };
}

// -- The import ---------------------------------------------------------------

/** ONNX operators whose engine op has the same meaning and no settings. */
const DIRECT: Record<string, string> = {
  Relu: "relu",
  Sigmoid: "sigmoid",
  HardSigmoid: "sigmoid",
  Tanh: "tanh",
  HardSwish: "hardswish",
  Softmax: "softmax",
  BatchNormalization: "batchnorm",
  GlobalAveragePool: "gap",
  Flatten: "flatten",
};

export function fromOnnx(bytes: Uint8Array, name?: string): Model {
  let graphBytes: Uint8Array | undefined;
  let opset: number | undefined;
  for (const f of fields(bytes)) {
    if (f.field === 7) graphBytes = f.bytes;
    else if (f.field === 8) {
      // OperatorSetIdProto: domain 1, version 2. The default domain is "" (or "ai.onnx").
      let domain = "";
      let version: number | undefined;
      for (const g of fields(f.bytes)) {
        if (g.field === 1) domain = text(g.bytes);
        else if (g.field === 2) version = g.int;
      }
      if ((domain === "" || domain === "ai.onnx") && version !== undefined) opset = version;
    }
  }
  if (!graphBytes) throw new OnnxError("This file has no graph, so it is not an ONNX model.");

  let graphName = "";
  const onnxNodes: OnnxNode[] = [];
  const constants = new Map<string, Tensor>();
  const graphInputs: ValueInfo[] = [];
  for (const f of fields(graphBytes)) {
    if (f.field === 1) onnxNodes.push(readNode(f.bytes));
    else if (f.field === 2) graphName = text(f.bytes);
    else if (f.field === 5) {
      const tensor = readTensor(f.bytes);
      constants.set(tensor.name, tensor);
    } else if (f.field === 11) graphInputs.push(readValueInfo(f.bytes));
  }

  const modelInput = graphInputs.find((input) => !constants.has(input.name));
  if (!modelInput) throw new OnnxError("This graph has no input that is not a weight.");
  const inputDims = modelInput.dims.slice(1);
  // A file may leave the image size open; fall back to the usual 224.
  const sizeOpen = inputDims.some((d) => d === null);
  const shape = inputDims.map((d) => d ?? 224);
  if (!shape.length) throw new OnnxError(`The input "${modelInput.name}" has no shape beyond the batch.`);
  const kind = shape.length === 3 ? "vision" : "tabular";

  const b = new Builder(name ?? (graphName || "ONNX model"), kind, "onnx");
  // Which engine node produces each ONNX tensor.
  const producer = new Map<string, string>();
  producer.set(modelInput.name, b.input(shape.join("x")));
  /** Tensors computed only from shapes and constants, such as the target of a dynamic Reshape. */
  const shapeTensors = new Set<string>();
  const nodeArgs = new Map<string, Record<string, ArgValue>>();
  const nodeOp = new Map<string, string>();
  const paths = new Map<string, string[]>();
  const notUnderstood: string[] = [];

  const isConstant = (tensor: string) => constants.has(tensor) || shapeTensors.has(tensor);
  const attr = (node: OnnxNode, key: string) => node.attrs.get(key);
  const intAttr = (node: OnnxNode, key: string, fallback: number) => attr(node, key)?.i ?? fallback;
  const firstInt = (node: OnnxNode, key: string, fallback: number) => attr(node, key)?.ints[0] ?? fallback;
  const scalar = (tensor: string | undefined): number | undefined => {
    const constant = tensor ? constants.get(tensor) : undefined;
    return constant ? values(constant, 1)[0] : undefined;
  };

  for (const node of onnxNodes) {
    if (node.op === "Constant") {
      for (const output of node.outputs) constants.set(output, constantValue(node));
      continue;
    }

    const present = node.inputs.filter(Boolean);
    const data = present.filter((tensor) => !isConstant(tensor));
    const sources = data.map((tensor) => producer.get(tensor)).filter((id): id is string => id !== undefined);
    const alias = (id: string) => node.outputs.forEach((output) => producer.set(output, id));

    // Shape reads a tensor's size, not its values, and anything computed only
    // from sizes and constants (the Gather, Unsqueeze, Concat chain PyTorch
    // writes for x.view(x.size(0), -1)) is a setting, not a layer.
    if (!data.length || node.op === "Shape" || node.op === "Size") {
      node.outputs.forEach((output) => shapeTensors.add(output));
      continue;
    }
    if (!sources.length) throw new OnnxError(`Node "${node.name || node.op}" reads "${data[0]}", which nothing before it produces.`);

    const add = (op: string, args: Record<string, ArgValue> = {}, inputs = sources.slice(0, 1), label?: string) => {
      const id = b.node(op, inputs, args, label);
      nodeArgs.set(id, args);
      nodeOp.set(id, op);
      const segments = node.name.split("/").filter(Boolean);
      if (segments.length > 1) paths.set(id, segments.slice(0, -1));
      alias(id);
    };
    const unknown = () => {
      if (!notUnderstood.includes(node.op)) notUnderstood.push(node.op);
      add("identity", {}, sources.slice(0, 1), node.op);
    };

    switch (node.op) {
      case "Dropout":
      case "Identity":
        alias(sources[0]);
        break;

      case "Conv": {
        const weight = constants.get(node.inputs[1]);
        if (!weight || weight.dims.length !== 4) {
          unknown();
          break;
        }
        const k = firstInt(node, "kernel_shape", weight.dims[2]);
        const autoPad = attr(node, "auto_pad")?.s ?? "NOTSET";
        const p = autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER" ? Math.floor(k / 2) : firstInt(node, "pads", 0);
        const groups = intAttr(node, "group", 1);
        add("conv2d", { out: weight.dims[0], k, s: firstInt(node, "strides", 1), p, ...(groups > 1 ? { groups } : {}), bias: Boolean(node.inputs[2]) });
        break;
      }

      case "MaxPool":
      case "AveragePool": {
        const k = firstInt(node, "kernel_shape", 1);
        add(node.op === "MaxPool" ? "maxpool" : "avgpool", { k, s: firstInt(node, "strides", 1), p: firstInt(node, "pads", 0) });
        break;
      }

      case "Clip": {
        // Opset 11 moved the bounds from attributes to inputs.
        const min = attr(node, "min")?.f ?? scalar(node.inputs[1]);
        const max = attr(node, "max")?.f ?? scalar(node.inputs[2]);
        add(min === 0 && max === 6 ? "relu6" : "relu");
        break;
      }

      case "Concat": {
        if (sources.length < 2) {
          alias(sources[0]);
          break;
        }
        // The engine's axes leave out the batch, ONNX's include it.
        const axis = intAttr(node, "axis", 1);
        add("concat", { axis: axis > 0 ? axis - 1 : axis }, sources);
        break;
      }

      case "Mul":
        // A product with a constant is a fixed scale; it keeps the shape and holds no layer.
        if (sources.length < 2) alias(sources[0]);
        else add("mul", {}, sources);
        break;

      case "Add": {
        if (sources.length >= 2) {
          add("add", {}, sources);
          break;
        }
        // A constant added to a layer without a bias is that layer's bias, split
        // out by exporters that write MatMul then Add instead of Gemm.
        const id = sources[0];
        const bias = constants.get(present.find((tensor) => constants.has(tensor)) ?? "");
        const args = nodeArgs.get(id);
        const size = bias ? bias.dims.reduce((a, d) => a * d, 1) : 0;
        if (args && args.bias === false && (nodeOp.get(id) === "linear" || nodeOp.get(id) === "conv2d") && size === args.out) args.bias = true;
        alias(id);
        break;
      }

      case "Gemm":
      case "MatMul": {
        const weight = constants.get(node.inputs[1]);
        if (!weight || weight.dims.length !== 2) {
          unknown();
          break;
        }
        // Gemm computes A x B', so with transB the weight is stored [out, in].
        const transposed = node.op === "Gemm" && intAttr(node, "transB", 0) === 1;
        add("linear", { out: transposed ? weight.dims[0] : weight.dims[1], bias: node.op === "Gemm" && Boolean(node.inputs[2]) });
        break;
      }

      case "Reshape": {
        const target = constants.get(node.inputs[1]);
        // A target worked out from Shape at run time is PyTorch's x.view(x.size(0), -1).
        if (!target ? shapeTensors.has(node.inputs[1]) : values(target).length === 2) add("flatten");
        else unknown();
        break;
      }

      default: {
        const direct = DIRECT[node.op];
        if (direct) add(direct);
        else unknown();
      }
    }
  }

  const { groups, groupOf } = buildGroups(paths);
  const model = b.build({ family: "ONNX", ...(opset !== undefined ? { opset } : {}), "ONNX ops": onnxNodes.length });
  model.groups = groups;
  for (const node of model.nodes) node.group = groupOf.get(node.id);
  if (notUnderstood.length) model.meta.note = `Not understood, shown as pass-through: ${notUnderstood.join(", ")}`;
  if (sizeOpen) model.meta.input = `size not fixed in the file, assumed ${shape.join("×")}`;
  return model;
}
