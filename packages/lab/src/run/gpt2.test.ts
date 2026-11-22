import { describe, expect, test } from "bun:test";
import { Gpt2Tokenizer } from "./bpe";
import { parseSafetensors } from "./gpt2";

describe("safetensors", () => {
  test("reads shapes and values, including at offsets that are not a multiple of 4", () => {
    const header = JSON.stringify({ a: { dtype: "F32", shape: [2], data_offsets: [0, 8] }, b: { dtype: "F32", shape: [1, 1], data_offsets: [8, 12] }, __metadata__: {} });
    const bytes = new TextEncoder().encode(header);
    const buffer = new ArrayBuffer(8 + bytes.length + 12);
    new DataView(buffer).setBigUint64(0, BigInt(bytes.length), true);
    new Uint8Array(buffer, 8).set(bytes);
    const values = new DataView(buffer, 8 + bytes.length);
    [1.5, -2, 3].forEach((v, i) => values.setFloat32(i * 4, v, true));
    const tensors = parseSafetensors(buffer);
    expect([...tensors.get("a")!.data]).toEqual([1.5, -2]);
    expect(tensors.get("b")!.shape).toEqual([1, 1]);
    expect(tensors.has("__metadata__")).toBe(false);
  });
});

describe("byte-level BPE", () => {
  test("merges by rank and round-trips bytes", () => {
    // A toy vocabulary: "Ġ" is GPT-2's stand-in for a space byte.
    const vocab = { h: 0, i: 1, "Ġ": 2, hi: 3, "Ġhi": 4 };
    const tokenizer = new Gpt2Tokenizer(vocab, "#version: 0.2\nh i\nĠ hi\n");
    expect(tokenizer.encode("hi hi")).toEqual([3, 4]);
    expect(tokenizer.decode([3, 4])).toBe("hi hi");
  });
});
