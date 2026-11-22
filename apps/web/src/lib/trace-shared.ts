"use client";

import type { TracedLayer } from "tessera-lab";
import { create } from "zustand";

// What the trace pane hands the 3D view beside it: the layer the trace is on,
// and the values every layer produced, so the 3D can glide to the layer and
// paint real activations onto its stages.

export interface TraceFocus {
  nodeId: string;
  copy: number;
  layer: TracedLayer;
  /** Labels of an attention layer's rows, for the panel. */
  rows?: string[];
}

interface TraceShared {
  focus: [TraceFocus | null, TraceFocus | null];
  /** Each node's output from the first copy of its block (node id to layer). */
  values: [Map<string, TracedLayer> | null, Map<string, TracedLayer> | null];
  setFocus(slot: 0 | 1, focus: TraceFocus | null): void;
  setValues(slot: 0 | 1, values: Map<string, TracedLayer> | null): void;
}

export const useTraceShared = create<TraceShared>((set, get) => ({
  focus: [null, null],
  values: [null, null],
  setFocus: (slot, focus) => {
    const next = [...get().focus] as TraceShared["focus"];
    next[slot] = focus;
    set({ focus: next });
  },
  setValues: (slot, values) => {
    const next = [...get().values] as TraceShared["values"];
    next[slot] = values;
    set({ values: next });
  },
}));
