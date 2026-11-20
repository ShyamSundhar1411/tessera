import { describe, expect, test } from "bun:test";
import { analyze } from "../analyze";
import { fromOnnx } from "./onnx";
import { fields, float32, pushFloats, pushInts, readVarint, text, Wire } from "./protobuf";

// -- A tiny protobuf writer, so tests can build messages field by field -----

const varint = (value: number): number[] => {
  const out: number[] = [];
  let n = BigInt.asUintN(64, BigInt(value));
  do {
    const byte = Number(n & 0x7fn);
    n >>= 7n;
    out.push(n ? byte | 0x80 : byte);
  } while (n);
  return out;
};
type Part = number[];
const key = (field: number, wire: number) => varint(field * 8 + wire);
const int = (field: number, value: number): Part => [...key(field, 0), ...varint(value)];
const bytes = (field: number, payload: Part | Uint8Array): Part => [...key(field, 2), ...varint(payload.length), ...payload];
const str = (field: number, value: string) => bytes(field, new TextEncoder().encode(value));
const msg = (field: number, ...parts: Part[]) => bytes(field, parts.flat());
const packed = (field: number, values: number[]) => bytes(field, values.flatMap(varint));
const f32 = (value: number): Part => {
  const buf = new Uint8Array(4);
  new DataView(buf.buffer).setFloat32(0, value, true);
  return [...buf];
};
const float = (field: number, value: number): Part => [...key(field, 5), ...f32(value)];

describe("protobuf reader", () => {
  test("reads varints, strings and skips fixed-width fields", () => {
    const buf = new Uint8Array([
      ...[0x08, 0x96, 0x01], // field 1, varint 150 (the example in the protobuf docs)
      ...[0x12, 0x07, ...new TextEncoder().encode("testing")], // field 2, "testing"
      ...[0x1d, ...f32(1.5)], // field 3, fixed32
      ...[0x21, 1, 2, 3, 4, 5, 6, 7, 8], // field 4, fixed64
      ...int(5, -1), // field 5, int64 -1, written as ten bytes
      ...packed(6, [3, 270, 86942]), // field 6, packed repeated
      ...int(6, 7), // field 6 again, unpacked
    ]);
    const all = [...fields(buf)];
    expect(all.map((f) => [f.field, f.wire])).toEqual([
      [1, Wire.Varint],
      [2, Wire.Bytes],
      [3, Wire.Fixed32],
      [4, Wire.Fixed64],
      [5, Wire.Varint],
      [6, Wire.Bytes],
      [6, Wire.Varint],
    ]);
    expect(all[0].int).toBe(150);
    expect(text(all[1].bytes)).toBe("testing");
    expect(float32(all[2].bytes)).toBe(1.5);
    expect(all[3].bytes.length).toBe(8);
    expect(all[4].int).toBe(-1);
    const ints: number[] = [];
    pushInts(ints, all[5]);
    pushInts(ints, all[6]);
    expect(ints).toEqual([3, 270, 86942, 7]);
  });

  test("returns views into the buffer, not copies", () => {
    const buf = new Uint8Array(bytes(9, new Array(1000).fill(7)));
    const [field] = [...fields(buf)];
    expect(field.bytes.buffer).toBe(buf.buffer);
    expect(field.bytes.length).toBe(1000);
  });

  test("reads packed floats and large varints", () => {
    const floats: number[] = [];
    pushFloats(floats, { field: 1, wire: Wire.Bytes, int: 0, bytes: new Uint8Array([...f32(0.5), ...f32(-2)]) });
    expect(floats).toEqual([0.5, -2]);
    expect(readVarint(new Uint8Array(varint(2 ** 40 + 3)), 0)).toEqual([2 ** 40 + 3, 6]);
  });

  test("rejects a truncated message", () => {
    expect(() => [...fields(new Uint8Array([0x12, 0x05, 0x01]))]).toThrow();
  });
});

// -- A hand-built ONNX graph in PyTorch's naming style -------------------------

const FLOAT = 1;
const INT64 = 7;
// TensorProto: dims 1, data_type 2, int64_data 7, name 8, raw_data 9.
const weight = (name: string, dims: number[]) => msg(5, packed(1, dims), int(2, FLOAT), str(8, name), bytes(9, new Array(4 * dims.reduce((a, b) => a * b, 1)).fill(0)));
const scalarInit = (name: string, value: number) => msg(5, int(2, FLOAT), str(8, name), bytes(9, f32(value)));
// AttributeProto: name 1, f 2, i 3, t 5, ints 8, type 20.
const attrInts = (name: string, values: number[]) => msg(5, str(1, name), packed(8, values), int(20, 7));
const attrInt = (name: string, value: number) => msg(5, str(1, name), int(3, value), int(20, 2));
const attrFloat = (name: string, value: number) => msg(5, str(1, name), float(2, value), int(20, 1));
// NodeProto: input 1, output 2, name 3, op_type 4, attribute 5.
const node = (op: string, name: string, inputs: string[], outputs: string[], ...attrs: Part[]) =>
  msg(1, ...inputs.map((i) => str(1, i)), ...outputs.map((o) => str(2, o)), str(3, name), str(4, op), ...attrs);
// ValueInfoProto name 1, type 2 -> tensor_type 1 -> (elem_type 1, shape 2) -> dim 1 -> dim_value 1 or dim_param 2.
const graphInput = (name: string, dims: (number | string)[]) => {
  const shape = msg(2, ...dims.map((d) => (typeof d === "number" ? msg(1, int(1, d)) : msg(1, str(2, d)))));
  return msg(11, str(1, name), msg(2, msg(1, int(1, FLOAT), shape)));
};

function tinyResNet(): Uint8Array {
  const graph = [
    node("Conv", "/layer1/layer1.0/conv1/Conv", ["x", "w1"], ["c1"], attrInts("kernel_shape", [3, 3]), attrInts("pads", [1, 1, 1, 1]), attrInts("strides", [1, 1])),
    node("Clip", "/layer1/layer1.0/act/Clip", ["c1", "clip_min", "clip_max"], ["a1"]),
    node("Conv", "/layer1/layer1.0/conv2/Conv", ["a1", "w2", "b2"], ["c2"], attrInts("kernel_shape", [3, 3]), attrInts("pads", [1, 1, 1, 1])),
    node("Add", "/layer1/layer1.0/Add", ["c2", "c1"], ["r1"]),
    node("Dropout", "/drop/Dropout", ["r1"], ["d1"]),
    node("GlobalAveragePool", "/pool/GlobalAveragePool", ["d1"], ["p1"]),
    node("Constant", "/Constant", [], ["target"], msg(5, str(1, "value"), msg(5, packed(1, [2]), int(2, INT64), packed(7, [1, -1])), int(20, 4))),
    node("Reshape", "/Reshape", ["p1", "target"], ["f1"]),
    node("MatMul", "/fc/MatMul", ["f1", "wm"], ["m1"]),
    node("Add", "/fc/Add", ["m1", "bm"], ["m2"]),
    node("Gemm", "/head/Gemm", ["m2", "wg"], ["g1"], attrInt("transB", 1)),
    node("LeakyRelu", "/head/LeakyRelu", ["g1"], ["l1"]),
    node("Clip", "/Clip", ["l1"], ["k1"], attrFloat("min", 0), attrFloat("max", 1)),
    node("Softmax", "/Softmax", ["k1"], ["out"]),
    weight("w1", [8, 3, 3, 3]),
    weight("w2", [8, 8, 3, 3]),
    weight("b2", [8]),
    weight("wm", [8, 10]),
    weight("bm", [10]),
    weight("wg", [5, 10]),
    scalarInit("clip_min", 0),
    scalarInit("clip_max", 6),
    str(2, "tiny"),
    // Initializers listed as inputs too, as older exporters do; they must not be taken for the image.
    graphInput("w1", [8, 3, 3, 3]),
    graphInput("x", ["N", 3, 32, 32]),
  ].flat();
  // ModelProto: ir_version 1, graph 7, opset_import 8 (domain 1, version 2).
  return new Uint8Array([...int(1, 8), ...msg(8, str(1, ""), int(2, 13)), ...bytes(7, graph)]);
}

describe("ONNX import", () => {
  test("maps a hand-built graph onto engine ops", () => {
    const model = fromOnnx(tinyResNet());
    const analysis = analyze(model);
    expect(analysis.errors).toEqual([]);
    expect(model.name).toBe("tiny");
    expect(model.kind).toBe("vision");
    expect(model.nodes[0].args.shape).toBe("3x32x32");
    expect(model.nodes.map((n) => n.op)).toEqual(["input", "conv2d", "relu6", "conv2d", "add", "gap", "flatten", "linear", "linear", "identity", "relu", "softmax"]);
    // conv1 216 (no bias), conv2 576 + 8, linear 80 + 10 (bias from the Add), Gemm 50 (no C).
    expect(analysis.params).toBe(216 + 584 + 90 + 50);
    expect(analysis.outputShape).toEqual([5]);
    expect(model.nodes.find((n) => n.op === "identity")!.label).toBe("LeakyRelu");
    expect(model.meta).toEqual({ family: "ONNX", opset: 13, "ONNX ops": 14, note: "Not understood, shown as pass-through: LeakyRelu" });
  });

  test("turns PyTorch module paths into nested groups", () => {
    const model = fromOnnx(tinyResNet());
    const stage = model.groups.find((g) => g.label === "layer1")!;
    const block = model.groups.find((g) => g.label === "layer1.0")!;
    expect(stage).toMatchObject({ kind: "layer", parent: undefined });
    expect(block).toMatchObject({ kind: "layer", parent: stage.id });
    // Modules that wrap a single op ("conv1", "act") are not worth a box of their own.
    // "fc" ends up holding one layer, since its Add is folded into the MatMul as a bias.
    expect(model.groups.map((g) => g.label)).toEqual(["layer1", "layer1.0", "head"]);
    const inBlock = model.nodes.filter((n) => n.group === block.id).map((n) => n.op);
    expect(inBlock).toEqual(["conv2d", "relu6", "conv2d", "add"]);
    expect(model.nodes.find((n) => n.op === "gap")!.group).toBeUndefined();
  });

  test("rejects bytes that are not a model", () => {
    expect(() => fromOnnx(new Uint8Array([0x08, 0x01]))).toThrow("no graph");
  });
});

// A real export: timm's mobilenetv2_100 written by torch.onnx at opset 16,
// from the ONNX model zoo (huggingface.co/onnxmodelzoo/mobilenetv2_100_Opset16).
const fixture = new Uint8Array(await Bun.file(new URL("../../fixtures/mobilenetv2_100_opset16.onnx", import.meta.url)).arrayBuffer());

describe("a real ONNX file: MobileNetV2", () => {
  const model = fromOnnx(fixture, "MobileNetV2");
  const analysis = analyze(model);

  test("analyses cleanly to 1000 classes", () => {
    expect(analysis.errors).toEqual([]);
    expect(analysis.outputShape).toEqual([1000]);
    expect(model.meta.note).toBeUndefined();
    expect(model.meta.opset).toBe(16);
  });

  test("parameter count matches the published 3,504,872", () => {
    const published = 3_504_872;
    expect(Math.abs(analysis.params - published) / published).toBeLessThan(0.01);
    // The exporter folds each batch norm into the convolution before it: the
    // norm's scale and shift (2 per channel) become one bias per channel. The
    // 17,056 parameters missing are exactly those channels, and the file
    // holds the remaining 3,487,816 weights, every one counted here.
    expect(analysis.params).toBe(published - 17_056);
  });

  test("keeps the layers and the block structure", () => {
    const count = (op: string) => model.nodes.filter((n) => n.op === op).length;
    expect([count("conv2d"), count("relu6"), count("add"), count("linear")]).toEqual([52, 35, 10, 1]);
    const block = model.groups.find((g) => g.label === "blocks.1.0")!;
    expect(block.kind).toBe("blocks");
    expect(model.groups.find((g) => g.id === block.parent)!.label).toBe("blocks.1");
    // Depthwise convolutions keep their groups.
    expect(model.nodes.some((n) => n.op === "conv2d" && n.args.groups === 96)).toBe(true);
  });
});
