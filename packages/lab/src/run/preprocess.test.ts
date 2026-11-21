import { describe, expect, test } from "bun:test";
import { cropRect, softmax, toNchw, topK } from "./preprocess";

describe("image preprocessing", () => {
  test("resize the short edge then centre crop", () => {
    // 512×256: the short edge goes to 256 (scale 1), the 224 crop sits in the middle.
    expect(cropRect(512, 256, { size: 224, resize: 256 })).toEqual({ sx: 144, sy: 16, sw: 224, sh: 224 });
    // 1000×1000 to 256 is scale 0.256, so the crop covers 875 source pixels.
    const r = cropRect(1000, 1000, { size: 224, resize: 256 });
    expect(r.sw).toBeCloseTo(875);
    expect(r.sx).toBeCloseTo(62.5);
  });

  test("normalises to channels-first, flipping to BGR when asked", () => {
    // Two pixels: pure red, then pure blue.
    const rgba = [255, 0, 0, 255, 0, 0, 255, 255];
    const spec = { mean: [0.5, 0.5, 0.5] as [number, number, number], std: [0.5, 0.5, 0.5] as [number, number, number] };
    expect(Array.from(toNchw(rgba.slice(0, 4), 1, { ...spec, channelOrder: "rgb" }))).toEqual([1, -1, -1]);
    expect(Array.from(toNchw(rgba.slice(4), 1, { ...spec, channelOrder: "rgb" }))).toEqual([-1, -1, 1]);
    expect(Array.from(toNchw(rgba.slice(4), 1, { ...spec, channelOrder: "bgr" }))).toEqual([1, -1, -1]);
  });

  test("softmax and top-k", () => {
    const p = softmax([1, 3, 2]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(topK(p, ["a", "b", "c"], 2).map((x) => x.label)).toEqual(["b", "c"]);
  });
});
