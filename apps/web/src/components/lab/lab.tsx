"use client";

import { ArrowLeftRight, Box, Pencil, Plus, Undo2, X } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import { formatCount } from "tessera-engine";
import { Mark } from "@/components/brand/mark";
import { useLab, type PanelName, type SlotIndex } from "@/lib/store";
import { cn } from "@/lib/utils";
import CodePanel from "./code-panel";
import ComparePanel from "./compare-panel";
import Graph2D from "./graph-2d";
import Inspector from "./inspector";
import Palette from "./palette";
import RunPanel from "./run-panel";
import SourceDialog from "./source-dialog";
import TextPanel from "./text-panel";
import TrainPanel from "./train-panel";

// three.js only runs in the browser.
const Graph3D = dynamic(() => import("./graph-3d"), { ssr: false, loading: () => <div className="grid h-full place-items-center text-[13px] text-muted">Loading 3D…</div> });

const PANELS: { id: PanelName; label: string }[] = [
  { id: "inspect", label: "Inspect" },
  { id: "compare", label: "Compare" },
  { id: "text", label: "Text" },
  { id: "code", label: "Export" },
  { id: "train", label: "Train" },
  { id: "run", label: "Run" },
];

function SlotHeader({ slot, onLoad }: { slot: SlotIndex; onLoad: () => void }) {
  const data = useLab((state) => state.slots[slot]);
  const active = useLab((state) => state.active);
  const clear = useLab((state) => state.clear);
  const setActive = useLab((state) => state.setActive);
  if (!data) return null;
  return (
    <div className={cn("flex items-center gap-2 border-b border-line bg-surface px-3 py-2", active === slot && "bg-surface-2")} onClick={() => setActive(slot)}>
      <span className={cn("grid size-5 flex-none place-items-center rounded text-[11px] font-semibold", slot ? "bg-[#f0569f] text-[#0b0d12]" : "bg-[#5fb4ff] text-[#0b0d12]")}>{slot ? "B" : "A"}</span>
      <button type="button" onClick={onLoad} className="min-w-0 truncate text-left text-[13px] font-medium hover:underline">
        {data.model.name}
      </button>
      <span className="ml-auto flex-none font-mono text-[11.5px] text-muted">
        {formatCount(data.analysis.params)}
        {data.analysis.activeParams !== data.analysis.params ? ` · ${formatCount(data.analysis.activeParams)} active` : ""}
      </span>
      {data.analysis.errors.length > 0 && <span className="flex-none rounded bg-danger/15 px-1.5 text-[11px] text-danger">{data.analysis.errors.length} issue{data.analysis.errors.length > 1 ? "s" : ""}</span>}
      <button type="button" onClick={(event) => { event.stopPropagation(); clear(slot); }} aria-label="Remove" className="grid size-6 flex-none place-items-center rounded text-muted hover:text-ink">
        <X className="size-3.5" />
      </button>
    </div>
  );
}

export default function Lab() {
  const slots = useLab((state) => state.slots);
  const view = useLab((state) => state.view);
  const panel = useLab((state) => state.panel);
  const editing = useLab((state) => state.editing);
  const active = useLab((state) => state.active);
  const setView = useLab((state) => state.setView);
  const setPanel = useLab((state) => state.setPanel);
  const setEditing = useLab((state) => state.setEditing);
  const swap = useLab((state) => state.swap);
  const undo = useLab((state) => state.undo);
  const [dialog, setDialog] = useState<SlotIndex | null>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "z" && !(event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement)) {
        event.preventDefault();
        undo(active);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, undo]);

  const shown = ([0, 1] as SlotIndex[]).filter((index) => slots[index]);
  const Graph = view === "3d" ? Graph3D : Graph2D;

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex h-12 flex-none items-center gap-2 border-b border-line px-3">
        <Link href="/" className="flex items-center gap-2 pr-2">
          <Mark className="size-6" />
          <span className="text-[15px] font-semibold tracking-tight">Tessera</span>
        </Link>
        <button type="button" onClick={() => setDialog(slots[0] ? 0 : 0)} className="rounded-lg border border-line px-2.5 py-1 text-[12.5px] hover:border-muted">
          Load A
        </button>
        <button type="button" onClick={() => setDialog(1)} className="flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-[12.5px] hover:border-muted">
          <Plus className="size-3.5" />
          {slots[1] ? "Load B" : "Add B to compare"}
        </button>
        {slots[1] && (
          <button type="button" onClick={swap} aria-label="Swap A and B" className="grid size-7 place-items-center rounded-lg border border-line hover:border-muted">
            <ArrowLeftRight className="size-3.5" />
          </button>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <button type="button" onClick={() => setEditing(!editing)} aria-pressed={editing} className={cn("flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-[12.5px] hover:border-muted", editing && "border-accent bg-accent/10 text-accent")}>
            <Pencil className="size-3.5" />
            Edit
          </button>
          <button type="button" onClick={() => undo(active)} aria-label="Undo" title="Undo (Ctrl/⌘ Z)" className="grid size-7 place-items-center rounded-lg border border-line hover:border-muted disabled:opacity-40" disabled={!slots[active]?.history.length}>
            <Undo2 className="size-3.5" />
          </button>
          <div className="flex rounded-lg border border-line p-0.5 text-[12.5px]">
            {(["2d", "3d"] as const).map((item) => (
              <button key={item} type="button" onClick={() => setView(item)} aria-pressed={view === item} className={cn("flex items-center gap-1 rounded-md px-2 py-0.5 text-muted", view === item && "bg-surface-3 text-ink")}>
                {item === "3d" && <Box className="size-3.5" />}
                {item.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {editing && view === "2d" && <Palette />}
        <main className={cn("grid min-h-0 min-w-0 flex-1", shown.length === 2 && "md:grid-cols-2")}>
          {shown.map((index) => (
            <section key={index} className={cn("flex min-h-0 flex-col", index === 1 && "border-l border-line")}>
              <SlotHeader slot={index} onLoad={() => setDialog(index)} />
              <div className="min-h-0 flex-1">
                <Graph slot={index} />
              </div>
            </section>
          ))}
          {shown.length === 0 && (
            <div className="grid place-items-center">
              <button type="button" onClick={() => setDialog(0)} className="rounded-lg bg-ink px-4 py-2 text-[14px] font-medium text-bg">
                Load an architecture
              </button>
            </div>
          )}
        </main>
        <aside className="flex w-[360px] flex-none flex-col border-l border-line bg-surface">
          <div className="flex flex-none gap-0.5 overflow-x-auto border-b border-line px-2 pt-2">
            {PANELS.map((item) => (
              <button key={item.id} type="button" onClick={() => setPanel(item.id)} className={cn("-mb-px border-b-2 border-transparent px-2 pb-2 text-[12.5px] whitespace-nowrap text-muted hover:text-ink", panel === item.id && "border-accent text-ink")}>
                {item.label}
              </button>
            ))}
          </div>
          <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto">
            {panel === "inspect" && <Inspector />}
            {panel === "compare" && <ComparePanel />}
            {panel === "text" && <TextPanel />}
            {panel === "code" && <CodePanel />}
            {panel === "train" && <TrainPanel />}
            {panel === "run" && <RunPanel />}
          </div>
        </aside>
      </div>
      {dialog !== null && <SourceDialog slot={dialog} onClose={() => setDialog(null)} />}
    </div>
  );
}
