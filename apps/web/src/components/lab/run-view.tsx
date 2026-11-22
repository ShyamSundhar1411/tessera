"use client";

import { ImagePlus, Play, Sparkles, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { CATALOG, type Model } from "tessera-engine";
import type { ClassifyResult, PretrainedEntry } from "tessera-lab";
import { useLab, type SlotIndex } from "@/lib/store";
import { isGpt2, trainedFor, useTrained } from "@/lib/trained";
import { cn } from "@/lib/utils";
import { DrawPad, Output, runTrace, showToken, usePreparedModel, type Inputs } from "./trace-view";
import { RealWeightsButton, WeightsBadge } from "./weights";

// Run the model that is on the canvas. Which weights it runs with depends on
// what exists for it: published ImageNet weights (ResNet, MobileNet...),
// OpenAI's GPT-2, or the weights the Train tab produced. Without any, it says
// so and points to the ways to get some.

const SLOT_COLOURS = ["#5fb4ff", "#f0569f"];

function useEntries() {
  const [entries, setEntries] = useState<PretrainedEntry[]>([]);
  useEffect(() => {
    void import("tessera-lab").then(({ PRETRAINED }) => setEntries(PRETRAINED));
  }, []);
  return entries;
}

/** The published ImageNet model for what the slot holds, matched through the catalog entry it was loaded from. */
function publishedFor(origin: string, entries: PretrainedEntry[]) {
  const id = CATALOG.find((entry) => entry.name === origin)?.id;
  return id ? entries.find((entry) => entry.catalogId === id) : undefined;
}

// -- Published image classifiers -------------------------------------------------------

function ImageRun({ entry }: { entry: PretrainedEntry }) {
  const [image, setImage] = useState<{ url: string; element: HTMLImageElement } | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (element: HTMLImageElement) => {
    setError(null);
    try {
      const { classify } = await import("tessera-lab");
      const output = await classify(entry, element, (progress) => {
        if (progress.stage === "download") setStatus(`Downloading the weights (${entry.sizeMB} MB, once)… ${progress.total ? Math.round(((progress.loaded ?? 0) / progress.total) * 100) : 0}%`);
        else setStatus(progress.stage === "load" ? "Starting the model…" : "Running…");
      });
      setResult(output);
      setStatus(`${(output.seconds * 1000).toFixed(0)} ms on ${output.backend}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus(null);
    }
  };

  const pick = (file: File | undefined) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const element = new Image();
    element.onload = () => {
      setImage((previous) => {
        if (previous) URL.revokeObjectURL(previous.url);
        return { url, element };
      });
      setResult(null);
      void run(element);
    };
    element.src = url;
  };

  return (
    <div className="grid gap-5 md:grid-cols-2">
      <label
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          pick(event.dataTransfer.files[0]);
        }}
        className="flex min-h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-line p-4 text-center text-[13px] text-muted transition hover:border-muted hover:bg-surface-2"
      >
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image.url} alt="Your picture" className="max-h-64 rounded-xl" />
        ) : (
          <>
            <ImagePlus className="size-7" />
            Drop a photo, or click to choose one
            <span className="text-[11.5px]">It stays in this browser.</span>
          </>
        )}
        <input type="file" accept="image/*" hidden onChange={(event) => pick(event.target.files?.[0])} />
      </label>
      <div className="space-y-3">
        <p className="text-[12.5px] text-muted">
          {entry.name}&apos;s published ImageNet weights{entry.precision === "uint8" ? ", quantised to 8 bits" : ""}, from{" "}
          <a href={`https://huggingface.co/${entry.repo}`} target="_blank" rel="noreferrer" className="text-ink underline">
            {entry.repo}
          </a>
          . It picks one of 1,000 everyday categories.
        </p>
        {status && <p className="font-mono text-[11.5px] text-muted">{status}</p>}
        {error && <p className="rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">{error}</p>}
        {result && (
          <ol className="space-y-2.5">
            {result.top5.map((prediction, i) => (
              <li key={prediction.index}>
                <div className="flex justify-between gap-2 text-[13px]">
                  <span className={cn("truncate", i === 0 && "font-semibold")}>{prediction.label}</span>
                  <span className="font-mono text-muted">{(prediction.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-surface-3">
                  <div className="h-full rounded-full bg-accent transition-[width] duration-700" style={{ width: `${prediction.probability * 100}%` }} />
                </div>
              </li>
            ))}
          </ol>
        )}
        {!result && !status && <p className="text-[12.5px] text-muted">Choose a picture to see what it thinks it is.</p>}
      </div>
    </div>
  );
}

// -- GPT-2 writing text ----------------------------------------------------------------

function GenerateRun({ slot, model }: { slot: SlotIndex; model: Model }) {
  const trained = trainedFor(slot, model, useTrained((state) => state.slots));
  const [prompt, setPrompt] = useState("The meaning of life is");
  const [written, setWritten] = useState<{ text: string; p: number; options: { text: string; p: number }[] }[]>([]);
  const [running, setRunning] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const stop = useRef(false);

  if (!trained?.tokenizer) {
    return (
      <div className="space-y-3 rounded-2xl border border-line bg-surface-2 p-5">
        <p className="text-[13px]">
          This is GPT-2 small. Its real weights can run right here: 124 million numbers, downloaded once and kept by your browser. Then it writes text one token at a time, and you can see the alternatives it considered at every step.
        </p>
        <RealWeightsButton slot={slot} model={model} weights="random" />
      </div>
    );
  }
  const tokenizer = trained.tokenizer;

  const generate = async () => {
    const lab = await import("tessera-lab");
    stop.current = false;
    setRunning(true);
    setWritten([]);
    const ids = tokenizer.encode(prompt);
    for (let step = 0; step < 40 && !stop.current; step++) {
      const logits = lab.nextTokenLogits(trained.graph, ids);
      let max = -Infinity;
      for (const v of logits) max = Math.max(max, v);
      const weights = Array.from(logits, (v) => Math.exp((v - max) / Math.max(temperature, 0.01)));
      const total = weights.reduce((a, b) => a + b, 0);
      const ranked = weights
        .map((w, i) => ({ i, p: w / total }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 40);
      // Sample from the top 40, as GPT-2's own samples did.
      const top = ranked.reduce((a, b) => a + b.p, 0);
      let r = Math.random() * top;
      const choice = ranked.find((option) => (r -= option.p) <= 0) ?? ranked[0];
      ids.push(choice.i);
      setWritten((current) => [...current, { text: tokenizer.decode([choice.i]), p: choice.p, options: ranked.slice(0, 5).map((o) => ({ text: showToken(tokenizer.decode([o.i])), p: o.p })) }]);
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (choice.i === 50256) break;
    }
    setRunning(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-[260px] flex-1 text-[12px] text-muted">
          Start it off
          <input value={prompt} onChange={(event) => setPrompt(event.target.value)} className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2 text-[14px] text-ink" />
        </label>
        <label className="w-40 text-[12px] text-muted">
          Temperature {temperature.toFixed(1)}
          <input type="range" min={0.1} max={1.5} step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} className="mt-2 w-full accent-[var(--accent)]" />
        </label>
        <button type="button" onClick={() => (running ? (stop.current = true) : void generate())} className="flex h-10 items-center gap-1.5 rounded-lg bg-ink px-4 text-[13px] font-medium text-bg">
          {running ? <Square className="size-3.5" /> : <Sparkles className="size-3.5" />}
          {running ? "Stop" : "Write"}
        </button>
      </div>
      <div className="rounded-2xl border border-line bg-surface p-5 text-[17px] leading-[1.9]">
        <span className="text-muted">{prompt}</span>
        {written.map((token, i) => (
          <span key={i} className="group relative rounded-sm" style={{ background: `color-mix(in oklab, var(--accent) ${Math.round((1 - token.p) * 45)}%, transparent)` }}>
            {token.text}
            <span className="pointer-events-none absolute top-full left-0 z-10 mt-1 hidden w-48 rounded-lg border border-line bg-surface p-2 text-[11.5px] leading-snug shadow-lg group-hover:block">
              {token.options.map((option, j) => (
                <span key={j} className="flex justify-between gap-2 font-mono">
                  <span className="truncate">{option.text}</span>
                  <span className="text-muted">{(option.p * 100).toFixed(1)}%</span>
                </span>
              ))}
            </span>
          </span>
        ))}
        {running && <span className="ml-0.5 inline-block h-5 w-0.5 animate-pulse bg-accent align-middle" />}
      </div>
      <p className="text-[11.5px] text-muted">Stronger highlight means the model was less sure of that token. Hover any token to see the five it weighed up.</p>
    </div>
  );
}

// -- Models trained in the Train tab ---------------------------------------------------

function TrainedRun({ slot, model }: { slot: SlotIndex; model: Model }) {
  const { value } = usePreparedModel(slot, model);
  const [text, setText] = useState("To be, or not to be, that is the question:");
  const [symbols, setSymbols] = useState("3a7f019c42e8");
  const [pad, setPad] = useState<HTMLCanvasElement | null>(null);
  const [padVersion, setPadVersion] = useState(0);
  const [result, setResult] = useState<ReturnType<typeof runTrace>>(null);
  const inputs = useMemo<Inputs>(() => ({ text, symbols, pad, padVersion }), [text, symbols, pad, padVersion]);
  useEffect(() => {
    if (!value) return;
    let live = true;
    const timer = window.setTimeout(() => void import("tessera-lab").then((lab) => live && setResult(runTrace(lab, value, inputs))), 100);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [value, inputs]);
  if (!value) return null;
  const kind = value.prepared.task.kind;
  const last = result?.layers[result.layers.length - 1];
  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div className="space-y-3">
        <p className="text-[12.5px] text-muted">{value.prepared.task.label}, with the weights from your last training run.</p>
        {kind === "image-classification" && <DrawPad setPad={setPad} onChange={() => setPadVersion((v) => v + 1)} />}
        {(kind === "language-modelling" || kind === "masked-tokens") && <input value={text} onChange={(event) => setText(event.target.value)} className="w-full rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[13px]" />}
        {(kind === "seq2seq" || kind === "sequence-classification") && <input value={symbols} onChange={(event) => setSymbols(event.target.value)} className="w-full rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[13px] tracking-widest" />}
      </div>
      <div className="rounded-2xl border border-line bg-surface p-4">{last && result && <Output layer={last} trace={result} />}</div>
    </div>
  );
}

// -- One slot --------------------------------------------------------------------------

function SlotRun({ slot, entries }: { slot: SlotIndex; entries: PretrainedEntry[] }) {
  const data = useLab((state) => state.slots[slot]);
  const setMode = useLab((state) => state.setMode);
  const trained = trainedFor(slot, data?.model, useTrained((state) => state.slots));
  if (!data) return null;
  const published = publishedFor(data.origin, entries);
  const gpt2 = isGpt2(data.model);
  const weights = trained?.kind ?? (published ? "pretrained" : "random");
  return (
    <section className="space-y-4">
      <header className="flex flex-wrap items-center gap-2">
        <span className="grid size-6 place-items-center rounded-md text-[12px] font-semibold text-[#0b0d12]" style={{ background: SLOT_COLOURS[slot] }}>
          {slot ? "B" : "A"}
        </span>
        <h2 className="text-[16px] font-semibold">{data.model.name}</h2>
        {(published || trained) && <WeightsBadge weights={weights} />}
      </header>
      {gpt2 ? (
        <GenerateRun slot={slot} model={data.model} />
      ) : published && trained?.kind !== "trained" ? (
        <ImageRun entry={published} />
      ) : trained ? (
        <TrainedRun slot={slot} model={data.model} />
      ) : (
        <div className="rounded-2xl border border-line bg-surface-2 p-5 text-[13px] leading-relaxed">
          <p className="font-medium">This architecture has no weights yet.</p>
          <p className="mt-1 text-muted">An architecture is only a plan until its numbers are learned. Train a small version of it here in a few seconds, then come back and run it on your own input.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" onClick={() => setMode("train")} className="flex items-center gap-1.5 rounded-lg bg-ink px-3 py-1.5 text-[12.5px] font-medium text-bg">
              <Play className="size-3.5" />
              Train it
            </button>
            <button type="button" onClick={() => setMode("trace")} className="rounded-lg border border-line px-3 py-1.5 text-[12.5px] hover:border-muted">
              Trace it with random weights
            </button>
          </div>
          <p className="mt-4 text-[12px] text-muted">Models with real published weights that run here: ResNet-18, ResNet-50, MobileNetV2 and the other ImageNet models in the catalog, and GPT-2.</p>
        </div>
      )}
    </section>
  );
}

export default function RunView() {
  const slots = useLab((state) => state.slots);
  const entries = useEntries();
  const shown = ([0, 1] as SlotIndex[]).filter((index) => slots[index]);
  return (
    <div className="mx-auto max-w-5xl space-y-10 px-5 py-8">
      {shown.map((slot) => (
        <SlotRun key={slot} slot={slot} entries={entries} />
      ))}
    </div>
  );
}
