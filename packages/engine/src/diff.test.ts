import { describe, expect, test } from "bun:test";
import { catalogModel } from "./catalog";
import { diff, metrics } from "./diff";

describe("diff", () => {
  test("adding one skip connection shows as one changed block", () => {
    const result = diff(catalogModel("cifar-cnn")!, catalogModel("cifar-cnn-skip")!);
    const changed = result.changes.filter((change) => change.kind !== "same");
    expect(changed).toHaveLength(1);
    expect(changed[0].kind).toBe("changed");
    expect(changed[0].detail).toContain("+1 add");
  });

  test("Llama vs Mistral: same blocks, different settings", () => {
    const result = diff(catalogModel("llama-3.1-8b")!, catalogModel("mistral-7b")!);
    const block = result.changes.find((change) => change.a?.key === "decoder")!;
    expect(block.kind).toBe("changed");
    expect(block.detail).toContain("window");
    const vocab = result.settings.find((row) => row.key === "vocab")!;
    expect(vocab.same).toBe(false);
  });

  test("Mistral vs Mixtral: the feed-forward becomes experts", () => {
    const result = diff(catalogModel("mistral-7b")!, catalogModel("mixtral-8x7b")!);
    const block = result.changes.find((change) => change.a?.key === "decoder")!;
    expect(block.kind).toBe("changed");
    expect(block.detail).toContain("-1 mlp");
    expect(block.detail).toContain("+1 moe");
  });

  test("KV cache per token", () => {
    // Llama 3.1 8B: 32 layers x 2 (K and V) x 8 heads x 128 dims x 2 bytes = 128 KiB.
    expect(metrics(catalogModel("llama-3.1-8b")!).kvBytesPerToken).toBe(131072);
    // DeepSeek-V3 caches the latent: 61 layers x (512 + 64) x 2 bytes.
    expect(metrics(catalogModel("deepseek-v3")!).kvBytesPerToken).toBe(61 * 576 * 2);
    expect(metrics(catalogModel("resnet50")!).kvBytesPerToken).toBeNull();
  });
});
