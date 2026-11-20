// A minimal reader for the protobuf wire format, enough to walk an ONNX file
// without a dependency. Protobuf messages are a flat list of (field number,
// wire type, value) records; nested messages are just length-delimited bytes
// read again with the same reader. Byte values are returned as views into the
// original buffer, never copies, so a 100 MB weight tensor costs nothing to
// step over.

// A plain object rather than an enum, since bundlers that compile one file at
// a time cannot inline `const enum`.
export const Wire = { Varint: 0, Fixed64: 1, Bytes: 2, Fixed32: 5 } as const;
export type Wire = (typeof Wire)[keyof typeof Wire];

export interface Field {
  field: number;
  wire: Wire;
  /** The value of a varint field, as a signed 64-bit integer (exact up to 2^53). */
  int: number;
  /** The raw bytes of a length-delimited, fixed32 or fixed64 field, as a view. */
  bytes: Uint8Array;
}

export class ProtobufError extends Error {}

const EMPTY = new Uint8Array(0);

/** Reads one varint starting at `pos`; returns the value and the position after it. */
export function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  // Up to 28 bits fit safely in plain number arithmetic.
  while (shift < 28) {
    if (pos >= buf.length) throw new ProtobufError("truncated varint");
    const byte = buf[pos++];
    result |= (byte & 0x7f) << shift;
    if (byte < 0x80) return [result >>> 0, pos];
    shift += 7;
  }
  // Longer ones (large or negative int64) go through BigInt so -1 stays -1.
  let big = BigInt(result >>> 0);
  let bigShift = BigInt(shift);
  for (let i = 0; i < 6; i++) {
    if (pos >= buf.length) throw new ProtobufError("truncated varint");
    const byte = buf[pos++];
    big |= BigInt(byte & 0x7f) << bigShift;
    if (byte < 0x80) return [Number(BigInt.asIntN(64, big)), pos];
    bigShift += 7n;
  }
  throw new ProtobufError("varint longer than 10 bytes");
}

/** Walks the fields of one message, in the order they are written. */
export function* fields(buf: Uint8Array): Generator<Field> {
  let pos = 0;
  while (pos < buf.length) {
    const [key, afterKey] = readVarint(buf, pos);
    pos = afterKey;
    const field = Math.floor(key / 8);
    const wire = (key & 7) as Wire;
    if (field === 0) throw new ProtobufError("field number 0 is not valid");
    switch (wire) {
      case Wire.Varint: {
        const [value, next] = readVarint(buf, pos);
        pos = next;
        yield { field, wire, int: value, bytes: EMPTY };
        break;
      }
      case Wire.Fixed64:
      case Wire.Fixed32: {
        const size = wire === Wire.Fixed64 ? 8 : 4;
        if (pos + size > buf.length) throw new ProtobufError("truncated fixed-width field");
        yield { field, wire, int: 0, bytes: buf.subarray(pos, pos + size) };
        pos += size;
        break;
      }
      case Wire.Bytes: {
        const [length, start] = readVarint(buf, pos);
        if (length < 0 || start + length > buf.length) throw new ProtobufError("length-delimited field runs past the end");
        yield { field, wire, int: 0, bytes: buf.subarray(start, start + length) };
        pos = start + length;
        break;
      }
      default:
        // Wire types 3 and 4 (groups) were deprecated before ONNX existed.
        throw new ProtobufError(`unsupported wire type ${wire}`);
    }
  }
}

const decoder = new TextDecoder();

export const text = (bytes: Uint8Array) => decoder.decode(bytes);

export const float32 = (bytes: Uint8Array) => new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);

/**
 * The values of a repeated integer field. Proto3 writes them packed (one
 * length-delimited run of varints), older writers one record per value; both
 * are accepted, so callers pass every record of the field here.
 */
export function pushInts(out: number[], f: Field): void {
  if (f.wire === Wire.Varint) {
    out.push(f.int);
    return;
  }
  if (f.wire !== Wire.Bytes) return;
  let pos = 0;
  while (pos < f.bytes.length) {
    const [value, next] = readVarint(f.bytes, pos);
    out.push(value);
    pos = next;
  }
}

/** The values of a repeated float field, packed or not. */
export function pushFloats(out: number[], f: Field): void {
  if (f.wire === Wire.Fixed32) {
    out.push(float32(f.bytes));
    return;
  }
  if (f.wire !== Wire.Bytes) return;
  const view = new DataView(f.bytes.buffer, f.bytes.byteOffset, f.bytes.length);
  for (let pos = 0; pos + 4 <= f.bytes.length; pos += 4) out.push(view.getFloat32(pos, true));
}
