import { describe, expect, test } from "bun:test";
import { analyze } from "./analyze";
import { CATALOG } from "./catalog";
import { diff } from "./diff";
import { shrinkLanguageModel } from "./shrink";

describe("shrinking language models keeps their wiring", () => {
  for (const entry of CATALOG.filter((item) => item.shelf === "Language models")) {
    test(entry.name, () => {
      const big = entry.build();
      const small = shrinkLanguageModel(big, { vocab: 50257 });
      const analysis = analyze(small);
      expect(analysis.errors).toEqual([]);
      expect(analysis.params).toBeLessThan(60e6);
      // Same blocks in the same order, only their sizes change.
      expect(diff(big, small).changes.every((change) => change.kind === "same" || change.kind === "changed")).toBe(true);
    });
  }

  test("grouped-query sharing survives: Llama keeps 4 query heads per key/value head", () => {
    const small = shrinkLanguageModel(CATALOG.find((item) => item.id === "llama-3.1-8b")!.build(), { heads: 8 });
    const attention = small.nodes.find((node) => node.op === "attention")!;
    expect(attention.args.heads).toBe(8);
    expect(attention.args.kv).toBe(2);
  });
});
