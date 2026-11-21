"use client";

import { analyze, CATALOG, diff, parseText, toText, type Analysis, type Diff, type Model, type ParseError } from "tessera-engine";
import { create } from "zustand";
import { topoSort } from "./edit";

// The lab's state: one or two architectures (A and B), what is selected,
// which blocks are open, and how they are shown. Editing either the graph or
// its text keeps the other in step.

export type SlotIndex = 0 | 1;
export type View = "2d" | "3d";
export type PanelName = "inspect" | "compare" | "text" | "code" | "train" | "run";

export interface Slot {
  model: Model;
  /** Where it came from, for the header: a catalog name, a Hugging Face repo, a file. */
  origin: string;
  text: string;
  analysis: Analysis;
  parseErrors: ParseError[];
  /** Blocks shown open on the canvas. */
  expanded: string[];
  /** Kept so edits made on the graph can be undone. */
  history: Model[];
}

interface LabState {
  slots: [Slot | null, Slot | null];
  active: SlotIndex;
  view: View;
  panel: PanelName;
  editing: boolean;
  selection: { slot: SlotIndex; id: string } | null;
  diff: Diff | null;
  load(slot: SlotIndex, model: Model, origin: string): void;
  clear(slot: SlotIndex): void;
  setText(slot: SlotIndex, text: string): void;
  edit(slot: SlotIndex, change: (model: Model) => Model): void;
  undo(slot: SlotIndex): void;
  select(selection: LabState["selection"]): void;
  toggleExpanded(slot: SlotIndex, groupId: string): void;
  setExpanded(slot: SlotIndex, groups: string[]): void;
  setActive(slot: SlotIndex): void;
  setView(view: View): void;
  setPanel(panel: PanelName): void;
  setEditing(editing: boolean): void;
  swap(): void;
}

/** Opens the first repeated block of a language model, so its residual connections show straight away. */
function defaultExpanded(model: Model): string[] {
  if (model.kind !== "language") return [];
  const first = model.groups.find((group) => !group.parent);
  return first ? [first.id] : [];
}

function makeSlot(model: Model, origin: string, previous?: Slot, keepText?: string, parseErrors: ParseError[] = []): Slot {
  return {
    model,
    origin,
    text: keepText ?? toText(model),
    analysis: analyze(model),
    parseErrors,
    expanded: previous ? previous.expanded.filter((id) => model.groups.some((group) => group.id === id)) : defaultExpanded(model),
    history: previous?.history ?? [],
  };
}

const withDiff = (slots: [Slot | null, Slot | null]) => ({ slots, diff: slots[0] && slots[1] ? diff(slots[0].model, slots[1].model) : null });

const starter = CATALOG.find((entry) => entry.id === "cifar-cnn")!;

export const useLab = create<LabState>((set, get) => ({
  ...withDiff([makeSlot(starter.build(), starter.name), null]),
  active: 0,
  view: "2d",
  panel: "inspect",
  editing: false,
  selection: null,

  load(slot, model, origin) {
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = makeSlot(model, origin);
    set({ ...withDiff(slots), active: slot, selection: null, panel: slots[0] && slots[1] ? "compare" : get().panel === "compare" ? "inspect" : get().panel });
  },

  clear(slot) {
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = null;
    if (slot === 0 && slots[1]) {
      slots[0] = slots[1];
      slots[1] = null;
    }
    set({ ...withDiff(slots), active: 0, selection: null, panel: get().panel === "compare" ? "inspect" : get().panel });
  },

  setText(slot, text) {
    const current = get().slots[slot];
    if (!current) return;
    const { model, errors } = parseText(text);
    const slots = [...get().slots] as [Slot | null, Slot | null];
    // Keep showing the last good graph while the text has mistakes.
    slots[slot] = errors.length ? { ...current, text, parseErrors: errors } : makeSlot({ ...model, meta: { ...current.model.meta, ...model.meta, family: current.model.meta.family ?? "Custom" } }, current.origin, current, text);
    set(withDiff(slots));
  },

  edit(slot, change) {
    const current = get().slots[slot];
    if (!current) return;
    const next = topoSort(change(current.model));
    if (next === current.model) return;
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = { ...makeSlot(next, current.origin, current), history: [...current.history.slice(-49), current.model] };
    set(withDiff(slots));
  },

  undo(slot) {
    const current = get().slots[slot];
    const previous = current?.history[current.history.length - 1];
    if (!current || !previous) return;
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = { ...makeSlot(previous, current.origin, current), history: current.history.slice(0, -1) };
    set(withDiff(slots));
  },

  select(selection) {
    set({ selection, ...(selection ? { active: selection.slot } : {}) });
  },

  toggleExpanded(slot, groupId) {
    const current = get().slots[slot];
    if (!current) return;
    const expanded = current.expanded.includes(groupId) ? current.expanded.filter((id) => id !== groupId) : [...current.expanded, groupId];
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = { ...current, expanded };
    set({ slots });
  },

  setExpanded(slot, groups) {
    const current = get().slots[slot];
    if (!current) return;
    const slots = [...get().slots] as [Slot | null, Slot | null];
    slots[slot] = { ...current, expanded: groups };
    set({ slots });
  },

  setActive: (active) => set({ active }),
  setView: (view) => set({ view }),
  setPanel: (panel) => set({ panel }),
  setEditing: (editing) => set({ editing }),
  swap() {
    const [a, b] = get().slots;
    if (!a || !b) return;
    set({ ...withDiff([b, a]), selection: null });
  },
}));
