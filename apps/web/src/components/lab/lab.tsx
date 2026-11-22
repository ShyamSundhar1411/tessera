"use client";

import { ArrowLeftRight, Box, Code2, Dumbbell, Network, Pencil, Play, Plus, Route, Sigma, Square, Undo2, X } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";
import { CATALOG, formatCount } from "tessera-engine";
import { Mark } from "@/components/brand/mark";
import { useLab, type Mode, type PanelName, type SlotIndex } from "@/lib/store";
import { cn } from "@/lib/utils";
import CodePanel from "./code-panel";
import ComparePanel from "./compare-panel";
import Graph2D from "./graph-2d";
import Inspector from "./inspector";
import MathsPanel from "./maths-panel";
import Palette from "./palette";
import SourceDialog from "./source-dialog";
import TextPanel from "./text-panel";
import TrainPanel from "./train-panel";

// The lab is split into modes, picked from the rail on the left: look at the
// architecture, trace an input through it, read its maths, train it, run it,
// export it. A and B stay loaded across all of them, and every mode shows
// them side by side when both are there.

const loading = (text: string) => <div className="grid h-full place-items-center text-[13px] text-muted">{text}</div>;
// three.js, KaTeX and tf.js only run in the browser.
const Graph3D = dynamic(() => import("./graph-3d"), { ssr: false, loading: () => loading("Loading 3D…") });
const TraceView = dynamic(() => import("./trace-view"), { ssr: false, loading: () => loading("Loading the trace…") });
const RunView = dynamic(() => import("./run-view"), { ssr: false, loading: () => loading("Loading…") });

const SLOT_COLOURS = ["#5fb4ff", "#f0569f"];

const MODES: { id: Mode; label: string; icon: typeof Network; title: string; text: string }[] = [
  { id: "design", label: "Architecture", icon: Network, title: "Architecture", text: "The layers and how they connect." },
  { id: "trace", label: "Trace", icon: Route, title: "Trace an input", text: "Follow one input through every layer: what each layer computes, written as maths with the real sizes, and the numbers it produces." },
  { id: "maths", label: "Maths", icon: Sigma, title: "The maths of the wiring", text: "Worked out from the architecture alone, no training: how far gradients travel, what each output can see, where the compute goes and what longer context costs." },
  { id: "train", label: "Train", icon: Dumbbell, title: "Train in your browser", text: "A small version of each model learns a task picked from its wiring, on your GPU, across several seeds so a difference between A and B can be told from luck." },
  { id: "run", label: "Run", icon: Play, title: "Run it", text: "Give the model on the canvas your own input, using real published weights where they exist or the weights you trained." },
  { id: "export", label: "Export", icon: Code2, title: "Export", text: "The same architecture as PyTorch code and a Colab notebook that trains and compares it." },
];

const SIDE_TABS: { id: PanelName; label: string }[] = [
  { id: "inspect", label: "Inspect" },
  { id: "compare", label: "Compare" },
  { id: "text", label: "Text" },
];

function ModelChip({ slot, onLoad }: { slot: SlotIndex; onLoad: () => void }) {
  const data = useLab((state) => state.slots[slot]);
  const clear = useLab((state) => state.clear);
  if (!data) return null;
  return (
    <span className="flex flex-none items-center rounded-full border border-line bg-surface pr-1 pl-1 text-[12.5px]">
      <button type="button" onClick={onLoad} title="Load a different model" className="flex min-w-0 items-center gap-2 rounded-full py-1 pr-2 pl-1 hover:bg-surface-2">
        <span className="grid size-5 flex-none place-items-center rounded-full text-[10.5px] font-semibold text-[#0b0d12]" style={{ background: SLOT_COLOURS[slot] }}>
          {slot ? "B" : "A"}
        </span>
        <span className="max-w-[110px] truncate font-medium sm:max-w-[180px]">{data.model.name}</span>
        <span className="font-mono text-[11px] text-muted">{formatCount(data.analysis.params)}</span>
        {data.analysis.errors.length > 0 && <span className="size-1.5 rounded-full bg-danger" title={`${data.analysis.errors.length} problem(s)`} />}
      </button>
      <button type="button" onClick={() => clear(slot)} aria-label={`Remove ${slot ? "B" : "A"}`} className="grid size-6 place-items-center rounded-full text-muted hover:bg-surface-2 hover:text-ink">
        <X className="size-3" />
      </button>
    </span>
  );
}

/** A full-width mode's heading: what it is for, in a sentence. */
function Page({ mode, children, wide }: { mode: Mode; children: ReactNode; wide?: boolean }) {
  const info = MODES.find((item) => item.id === mode)!;
  return (
    <div className="scroll-quiet h-full overflow-y-auto">
      <div className={cn("mx-auto px-5 py-8", wide ? "max-w-5xl" : "max-w-3xl")}>
        <h1 className="text-[22px] font-semibold tracking-tight">{info.title}</h1>
        <p className="mt-1 max-w-2xl text-[13.5px] leading-relaxed text-muted">{info.text}</p>
        <div className="mt-6 overflow-hidden rounded-2xl border border-line bg-surface">{children}</div>
      </div>
    </div>
  );
}

function DesignTools() {
  const view = useLab((state) => state.view);
  const editing = useLab((state) => state.editing);
  const active = useLab((state) => state.active);
  const slots = useLab((state) => state.slots);
  const setView = useLab((state) => state.setView);
  const setEditing = useLab((state) => state.setEditing);
  const undo = useLab((state) => state.undo);
  return (
    <div className="flex items-center gap-1.5">
      <button type="button" onClick={() => setEditing(!editing)} aria-pressed={editing} className={cn("flex h-8 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[12.5px] hover:border-muted", editing && "border-accent bg-accent/10 text-accent")}>
        <Pencil className="size-3.5" />
        <span className="hidden sm:inline">Edit</span>
      </button>
      <button type="button" onClick={() => undo(active)} aria-label="Undo" title="Undo (Ctrl/⌘ Z)" className="grid size-8 place-items-center rounded-lg border border-line hover:border-muted disabled:opacity-40" disabled={!slots[active]?.history.length}>
        <Undo2 className="size-3.5" />
      </button>
      <div className="flex rounded-lg border border-line p-0.5 text-[12.5px]">
        {(["2d", "3d"] as const).map((item) => (
          <button key={item} type="button" onClick={() => setView(item)} aria-pressed={view === item} className={cn("flex h-7 items-center gap-1 rounded-md px-2.5 text-muted", view === item && "bg-surface-3 text-ink")}>
            {item === "3d" ? <Box className="size-3.5" /> : <Square className="size-3.5" />}
            {item.toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  );
}

function Design({ onLoad }: { onLoad: (slot: SlotIndex) => void }) {
  const slots = useLab((state) => state.slots);
  const view = useLab((state) => state.view);
  const panel = useLab((state) => state.panel);
  const editing = useLab((state) => state.editing);
  const active = useLab((state) => state.active);
  const setPanel = useLab((state) => state.setPanel);
  const setActive = useLab((state) => state.setActive);
  const shown = ([0, 1] as SlotIndex[]).filter((index) => slots[index]);
  const Graph = view === "3d" ? Graph3D : Graph2D;
  const tabs = SIDE_TABS.filter((tab) => tab.id !== "compare" || shown.length === 2);
  return (
    <div className="flex h-full min-h-0 flex-col lg:flex-row">
      {editing && view === "2d" && (
        <div className="hidden lg:block">
          <Palette />
        </div>
      )}
      <div className={cn("grid min-h-0 min-w-0 flex-1", shown.length === 2 && "grid-rows-2 md:grid-cols-2 md:grid-rows-1")}>
        {shown.map((index) => (
          <section key={index} onPointerDown={() => setActive(index)} className={cn("relative min-h-0", index === 1 && "border-t border-line md:border-t-0 md:border-l")}>
            {shown.length === 2 && (
              <button type="button" onClick={() => onLoad(index)} className={cn("absolute top-3 left-3 z-10 flex items-center gap-1.5 rounded-full border bg-surface/90 px-2.5 py-1 text-[12px] backdrop-blur", active === index ? "border-muted" : "border-line")}>
                <span className="size-2 rounded-full" style={{ background: SLOT_COLOURS[index] }} />
                {slots[index]!.model.name}
              </button>
            )}
            <Graph slot={index} />
          </section>
        ))}
        {shown.length === 0 && (
          <div className="grid place-items-center">
            <button type="button" onClick={() => onLoad(0)} className="rounded-lg bg-ink px-4 py-2 text-[14px] font-medium text-bg">
              Load an architecture
            </button>
          </div>
        )}
      </div>
      {/* On phones the side panel sits under the canvas; on wide screens, beside it. */}
      <aside className="flex h-[42dvh] w-full flex-none flex-col border-t border-line bg-surface lg:h-auto lg:w-[340px] lg:border-t-0 lg:border-l">
        <div className="flex flex-none gap-1 border-b border-line p-2">
          {tabs.map((item) => (
            <button key={item.id} type="button" onClick={() => setPanel(item.id)} className={cn("flex-1 rounded-md py-1.5 text-[12.5px] text-muted hover:text-ink", panel === item.id && "bg-surface-3 text-ink")}>
              {item.label}
            </button>
          ))}
        </div>
        <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto">
          {(panel === "inspect" || (panel === "compare" && shown.length < 2)) && <Inspector />}
          {panel === "compare" && shown.length === 2 && <ComparePanel />}
          {panel === "text" && <TextPanel />}
        </div>
      </aside>
    </div>
  );
}

export default function Lab() {
  const slots = useLab((state) => state.slots);
  const mode = useLab((state) => state.mode);
  const active = useLab((state) => state.active);
  const setMode = useLab((state) => state.setMode);
  const setView = useLab((state) => state.setView);
  const swap = useLab((state) => state.swap);
  const undo = useLab((state) => state.undo);
  const [dialog, setDialog] = useState<SlotIndex | null>(null);

  // Links like /lab/?a=mistral-7b&b=mixtral-8x7b&mode=trace open straight into a comparison.
  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const load = useLab.getState().load;
    (["a", "b"] as const).forEach((key, index) => {
      const entry = CATALOG.find((item) => item.id === query.get(key));
      if (entry) load(index as SlotIndex, entry.build(), entry.name);
    });
    const view = query.get("view");
    if (view === "3d") setView("3d");
    const wanted = query.get("mode") ?? (view === "trace" ? "trace" : null);
    if (MODES.some((item) => item.id === wanted)) setMode(wanted as Mode);
  }, [setMode, setView]);

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

  const loaded = slots.some(Boolean);

  return (
    <div className="flex h-dvh flex-col">
      <header className="scroll-quiet flex h-14 flex-none items-center gap-3 overflow-x-auto border-b border-line px-3 whitespace-nowrap">
        <Link href="/" className="flex flex-none items-center gap-2 pr-1">
          <Mark className="size-6" />
          <span className="hidden text-[15px] font-semibold tracking-tight sm:inline">Tessera</span>
        </Link>
        <span className="h-6 w-px flex-none bg-line" />
        {slots[0] ? (
          <ModelChip slot={0} onLoad={() => setDialog(0)} />
        ) : (
          <button type="button" onClick={() => setDialog(0)} className="rounded-full border border-line px-3 py-1 text-[12.5px] hover:border-muted">
            Load a model
          </button>
        )}
        {slots[1] ? (
          <>
            <button type="button" onClick={swap} aria-label="Swap A and B" title="Swap A and B" className="grid size-7 flex-none place-items-center rounded-full text-muted hover:bg-surface-2 hover:text-ink">
              <ArrowLeftRight className="size-3.5" />
            </button>
            <ModelChip slot={1} onLoad={() => setDialog(1)} />
          </>
        ) : (
          slots[0] && (
            <button type="button" onClick={() => setDialog(1)} className="flex flex-none items-center gap-1 rounded-full border border-dashed border-line px-3 py-1 text-[12.5px] text-muted hover:border-muted hover:text-ink">
              <Plus className="size-3.5" />
              Compare with another
            </button>
          )
        )}
        <div className="ml-auto flex-none pl-2">{mode === "design" && loaded && <DesignTools />}</div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col-reverse md:flex-row">
        {/* The modes: a rail on the left on wide screens, a bar along the bottom on phones. */}
        <nav className="flex flex-none justify-around border-t border-line bg-surface md:w-[76px] md:flex-col md:justify-start md:gap-1 md:border-t-0 md:border-r md:py-3">
          {MODES.map((item) => {
            const Icon = item.icon;
            const on = mode === item.id;
            return (
              <button key={item.id} type="button" onClick={() => setMode(item.id)} aria-current={on ? "page" : undefined} className="group flex flex-1 flex-col items-center gap-1 px-1 py-2 text-[10.5px] md:flex-none">
                <span className={cn("grid h-8 w-11 place-items-center rounded-xl transition-colors", on ? "bg-accent/15 text-accent" : "text-muted group-hover:bg-surface-2 group-hover:text-ink")}>
                  <Icon className="size-[18px]" />
                </span>
                <span className={cn(on ? "text-ink" : "text-muted")}>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <main className="min-h-0 min-w-0 flex-1">
          {!loaded && mode !== "design" ? (
            <div className="grid h-full place-items-center">
              <button type="button" onClick={() => setDialog(0)} className="rounded-lg bg-ink px-4 py-2 text-[14px] font-medium text-bg">
                Load an architecture
              </button>
            </div>
          ) : (
            <>
              {mode === "design" && <Design onLoad={setDialog} />}
              {mode === "trace" && <TraceView />}
              {mode === "maths" && (
                <Page mode="maths">
                  <MathsPanel />
                </Page>
              )}
              {mode === "train" && (
                <Page mode="train">
                  <TrainPanel />
                </Page>
              )}
              {mode === "run" && (
                <div className="scroll-quiet h-full overflow-y-auto">
                  <div className="mx-auto max-w-5xl px-5 pt-8">
                    <h1 className="text-[22px] font-semibold tracking-tight">{MODES[4].title}</h1>
                    <p className="mt-1 max-w-2xl text-[13.5px] leading-relaxed text-muted">{MODES[4].text}</p>
                  </div>
                  <RunView />
                </div>
              )}
              {mode === "export" && (
                <Page mode="export" wide>
                  <CodePanel />
                </Page>
              )}
            </>
          )}
        </main>
      </div>
      {dialog !== null && <SourceDialog slot={dialog} onClose={() => setDialog(null)} />}
    </div>
  );
}
