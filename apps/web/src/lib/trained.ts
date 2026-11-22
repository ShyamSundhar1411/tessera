"use client";

import type { Model } from "tessera-engine";
import type { GraphModel, PreparedModel } from "tessera-lab";
import { create } from "zustand";

// Models the Train tab finished training, kept so the trace view can run the
// trained weights instead of fresh ones. Keyed by slot, and only valid while
// that slot still holds the same model object.

export interface Trained {
  /** The slot's model when training started; an edit makes a new object and retires this. */
  source: Model;
  graph: GraphModel;
  prepared: PreparedModel;
}

interface TrainedState {
  slots: [Trained | null, Trained | null];
  keep(slot: 0 | 1, trained: Trained): void;
}

export const useTrained = create<TrainedState>((set, get) => ({
  slots: [null, null],
  keep: (slot, trained) => {
    const slots = [...get().slots] as TrainedState["slots"];
    slots[slot]?.graph.dispose();
    slots[slot] = trained;
    set({ slots });
  },
}));

/** The trained model for a slot, if it still matches what the slot shows. */
export function trainedFor(slot: 0 | 1, model: Model | undefined, slots: TrainedState["slots"]): Trained | null {
  const trained = slots[slot];
  return trained && model && trained.source === model ? trained : null;
}
