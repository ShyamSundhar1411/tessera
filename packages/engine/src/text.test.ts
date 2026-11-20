import { describe, expect, test } from "bun:test";
import { analyze } from "./analyze";
import { CATALOG } from "./catalog";
import { parseText, toText } from "./text";

describe("text format", () => {
  test("a small network with a named skip", () => {
    const { model, errors } = parseText(`
model "Skip demo"
input 3x32x32
stem = conv 32 k3 -> bn -> relu
y = conv 32 -> bn -> relu -> conv 32 -> bn
join = add(y, stem) -> relu
pool -> flatten -> linear 10
`);
    expect(errors).toEqual([]);
    const analysis = analyze(model);
    expect(analysis.errors).toEqual([]);
    expect(analysis.outputShape).toEqual([10]);
    expect(model.nodes.find((node) => node.op === "add")!.inputs).toEqual(["y", "stem"]);
  });

  test("errors point at their line", () => {
    const { errors } = parseText("input 3x32x32\nconv 16\nfrobnicate 3\nadd(a, nowhere)");
    expect(errors.map((error) => error.line)).toEqual([3, 4, 4]);
  });

  test("a shape mismatch is caught by the analysis, on the right line", () => {
    const { model } = parseText("input 3x32x32\na = conv 16\nb = @input -> conv 32\nadd(a, b)");
    const analysis = analyze(model);
    expect(analysis.errors[0].line).toBe(4);
    expect(analysis.errors[0].message).toContain("matching shapes");
  });

  test("repeated blocks with @in", () => {
    const { model, errors } = parseText(`
input tokens 128
embed 64 vocab=1000
block "Layer" x4 {
  h = norm -> attn 4
  mid = add(h, @in)
  f = norm -> mlp 256
  add(f, mid)
}
norm -> head 1000
`);
    expect(errors).toEqual([]);
    const analysis = analyze(model);
    expect(analysis.errors).toEqual([]);
    expect(model.groups[0].repeat).toBe(4);
    expect(analysis.outputShape).toEqual([128, 1000]);
  });

  describe("every catalog model survives a round trip through text", () => {
    for (const entry of CATALOG) {
      test(entry.name, () => {
        const model = entry.build();
        const { model: back, errors } = parseText(toText(model));
        expect(errors).toEqual([]);
        const a = analyze(model);
        const b = analyze(back);
        expect(b.errors).toEqual([]);
        expect(b.params).toBe(a.params);
        expect(b.flops).toBe(a.flops);
        expect(back.nodes.length).toBe(model.nodes.length);
      });
    }
  });
});
