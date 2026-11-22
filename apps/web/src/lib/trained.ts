"use client";

import type { Model } from "tessera-engine";
import type { Gpt2Progress, Gpt2Tokenizer, GraphModel, PreparedModel } from "tessera-lab";
import { create } from "zustand";

// Weights for the model in each slot that the trace and run views can use:
// a model the Train tab finished training, or real published weights (GPT-2).
// Only valid while that slot still holds the same model object.

export interface Trained {
  /** The slot's model when the weights arrived; an edit makes a new object and retires this. */
  source: Model;
  graph: GraphModel;
  prepared: PreparedModel;
  kind: "trained" | "pretrained";
  /** For real language models: their own tokenizer instead of characters. */
  tokenizer?: Gpt2Tokenizer;
}

interface TrainedState {
  slots: [Trained | null, Trained | null];
  keep(slot: 0 | 1, trained: Trained): void;
}

export const useTrained = create<TrainedState>((set, get) => ({
  slots: [null, null],
  keep: (slot, trained) => {
    const slots = [...get().slots] as TrainedState["slots"];
    if (slots[slot] && slots[slot]!.graph !== trained.graph) slots[slot]!.graph.dispose();
    slots[slot] = trained;
    set({ slots });
  },
}));

/** The weights for a slot, if they still match what the slot shows. */
export function trainedFor(slot: 0 | 1, model: Model | undefined, slots: TrainedState["slots"]): Trained | null {
  const trained = slots[slot];
  return trained && model && trained.source === model ? trained : null;
}

/** GPT-2 small, the only language model whose real weights the browser loads. */
export function isGpt2(model: Model | undefined) {
  if (!model) return false;
  const embedding = model.nodes.find((node) => node.op === "embedding");
  const block = model.groups.find((group) => (group.repeat ?? 1) > 1);
  const attention = model.nodes.find((node) => node.op === "attention");
  return String(model.meta.family ?? "").toLowerCase() === "gpt2" && Number(embedding?.args.vocab) === 50257 && Number(embedding?.args.dim) === 768 && block?.repeat === 12 && Number(attention?.args.heads) === 12;
}

/** Downloads OpenAI's GPT-2 weights (once; the browser caches them) and attaches them to a slot. */
export async function loadRealGpt2(slot: 0 | 1, model: Model, onProgress: (progress: Gpt2Progress) => void) {
  const lab = await import("tessera-lab");
  const gpt2 = await lab.loadGpt2(64, onProgress);
  const prepared: PreparedModel = {
    ok: true,
    task: { kind: "language-modelling", label: "Next-token prediction with OpenAI's trained GPT-2", accuracyLabel: "next-token accuracy", source: lab.GPT2_FILES.weights, classes: 50257, vocab: 50257, seqLen: 64, tokenLevel: true, key: "gpt2-real" },
    model: gpt2.model,
    description: "OpenAI's GPT-2 (124M) with its published weights, full size, reading up to 64 tokens.",
    params: gpt2.graph.paramCount,
    originalParams: gpt2.graph.paramCount,
    roles: { [gpt2.graph.inputs[0].id]: "tokens" },
  };
  useTrained.getState().keep(slot, { source: model, graph: gpt2.graph, prepared, kind: "pretrained", tokenizer: gpt2.tokenizer });
}
