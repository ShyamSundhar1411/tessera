"use client";

import { ExternalLink, FileUp, Search, X } from "lucide-react";
import { useEffect, useState } from "react";
import { CATALOG, PAPER_TIMELINE, fetchConfig, fromHuggingFace, fromOnnx, parseText, type Model } from "tessera-engine";
import { useLab, type SlotIndex } from "@/lib/store";
import { cn } from "@/lib/utils";

type Tab = "catalog" | "papers" | "huggingface" | "config" | "onnx" | "blank";
const TABS: { id: Tab; label: string }[] = [
  { id: "catalog", label: "Catalog" },
  { id: "papers", label: "Papers" },
  { id: "huggingface", label: "Hugging Face" },
  { id: "config", label: "Paste config.json" },
  { id: "onnx", label: "ONNX file" },
  { id: "blank", label: "Start empty" },
];

/** The papers everything else builds on, shown first. */
const CLASSICS = ["transformer-base", "resnet50", "alexnet", "gpt2", "bert-base", "vit-b16", "unet", "lenet5"];

const SUGGESTED = ["Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M", "microsoft/Phi-3.5-mini-instruct", "mistralai/Mistral-Nemo-Base-2407", "Qwen/Qwen3-4B", "allenai/OLMo-2-1124-7B"];

const BLANK = {
  vision: `model "My CNN"
input 3x32x32
stem = conv 32 k3 -> bn -> relu
y = conv 32 -> bn -> relu -> conv 32 -> bn
join = add(y, stem) -> relu
maxpool 2 -> conv 64 -> bn -> relu -> gap -> flatten -> linear 10
`,
  language: `model "My tiny transformer"
input tokens 256
embedding 256 vocab=50257
block "Decoder layer" x4 {
  h = rmsnorm -> attention 4 kv=2
  mid = add(h, @in)
  f = rmsnorm -> mlp 768
  add(f, mid)
}
rmsnorm -> lmhead 50257 tied
`,
};

export default function SourceDialog({ slot, onClose }: { slot: SlotIndex; onClose: () => void }) {
  const load = useLab((state) => state.load);
  const [tab, setTab] = useState<Tab>("catalog");
  const [repo, setRepo] = useState("");
  const [pasted, setPasted] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const finish = (model: Model, origin: string) => {
    load(slot, model, origin);
    onClose();
  };
  const attempt = async (work: () => Promise<void> | void) => {
    setBusy(true);
    setError(null);
    try {
      await work();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };
  const shelves = [...new Set(CATALOG.map((entry) => entry.shelf))];

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={`Load architecture ${slot ? "B" : "A"}`} className="flex h-[min(86dvh,760px)] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <header className="flex flex-none items-center justify-between border-b border-line px-5 py-3">
          <h2 className="text-[15px]">Load architecture {slot ? "B" : "A"}</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="grid size-8 place-items-center rounded-md text-muted hover:bg-surface-2 hover:text-ink">
            <X className="size-4" />
          </button>
        </header>
        <div className="scroll-quiet flex flex-none gap-1 overflow-x-auto border-b border-line px-3 py-2">
          {TABS.map((item) => (
            <button key={item.id} type="button" onClick={() => setTab(item.id)} className={cn("flex-none rounded-md px-2.5 py-1 text-[13px] whitespace-nowrap text-muted hover:text-ink", tab === item.id && "bg-surface-3 text-ink")}>
              {item.label}
            </button>
          ))}
        </div>
        <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto p-5">
          {tab === "catalog" && (
            <section className="mb-6">
              <h3 className="mb-2 flex items-baseline justify-between text-[12.5px] text-muted">
                The classics
                <button type="button" onClick={() => setTab("papers")} className="text-accent hover:underline">
                  All {PAPER_TIMELINE.length} papers on a timeline
                </button>
              </h3>
              <ul className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {CLASSICS.map((id) => CATALOG.find((entry) => entry.id === id)!).map((entry) => (
                  <li key={entry.id}>
                    <button type="button" onClick={() => finish(entry.build(), entry.name)} className="h-full w-full rounded-xl border border-line bg-gradient-to-b from-surface-2 to-surface p-3 text-left transition hover:border-accent">
                      <span className="font-mono text-[11px] text-accent">{entry.paper?.year}</span>
                      <span className="mt-0.5 block text-[13.5px] font-semibold">{entry.name}</span>
                      <span className="mt-1 block text-[11.5px] leading-snug text-muted">{entry.paper?.title}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {tab === "catalog" &&
            shelves.map((shelf) => (
              <section key={shelf} className="mb-5 last:mb-0">
                <h3 className="mb-2 text-[12.5px] text-muted">{shelf}</h3>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {CATALOG.filter((entry) => entry.shelf === shelf).map((entry) => (
                    <li key={entry.id}>
                      <button type="button" onClick={() => finish(entry.build(), entry.name)} className="h-full w-full rounded-xl border border-line p-3 text-left transition hover:border-muted hover:bg-surface-2">
                        <span className="flex items-baseline justify-between gap-2">
                          <span className="text-[13.5px] font-medium">{entry.name}</span>
                          {entry.published && <span className="font-mono text-[11px] text-muted">{entry.published}</span>}
                        </span>
                        <span className="mt-1 block text-[12.5px] leading-snug text-muted">{entry.blurb}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ))}

          {tab === "papers" && (
            <ol className="relative ml-2 border-l border-line">
              {PAPER_TIMELINE.map((entry, index) => {
                const paper = entry.paper!;
                const newYear = index === 0 || PAPER_TIMELINE[index - 1].paper!.year !== paper.year;
                return (
                  <li key={entry.id} className="relative pb-4 pl-5 last:pb-0">
                    <span className={cn("absolute top-1.5 -left-[5px] size-2.5 rounded-full border-2 border-surface", newYear ? "bg-accent" : "bg-muted")} />
                    {newYear && <p className="mb-1 font-mono text-[12px] font-semibold text-accent">{paper.year}</p>}
                    <div className="rounded-xl border border-line p-3">
                      <div className="flex items-start gap-2">
                        <div className="min-w-0 flex-1">
                          <a href={paper.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[13.5px] font-medium hover:underline">
                            {paper.title}
                            <ExternalLink className="size-3 flex-none text-muted" />
                          </a>
                          <p className="mt-0.5 text-[11.5px] text-muted">{paper.authors}</p>
                        </div>
                        <button type="button" onClick={() => finish(entry.build(), entry.name)} className="flex-none rounded-lg border border-line px-2.5 py-1 text-[12px] hover:border-muted">
                          Load {entry.name}
                        </button>
                      </div>
                      <p className="mt-2 text-[12.5px] leading-snug">{paper.idea}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}

          {tab === "huggingface" && (
            <div className="space-y-3">
              <p className="text-[13px] leading-relaxed text-muted">Any open model whose config.json describes a decoder (Llama, Mistral, Mixtral, Qwen, Phi, Gemma, DeepSeek, OLMo, GPT-2 and their many fine-tunes) or a BERT-style encoder (BERT, RoBERTa, DistilBERT, ELECTRA). Gated models (Llama, Gemma) need their config pasted instead.</p>
              <form
                className="flex gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  void attempt(async () => finish(fromHuggingFace(await fetchConfig(repo), repo.trim()), repo.trim()));
                }}
              >
                <label className="relative flex-1">
                  <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted" />
                  <input value={repo} onChange={(event) => setRepo(event.target.value)} placeholder="owner/model, e.g. Qwen/Qwen2.5-0.5B" className="w-full rounded-lg border border-line bg-bg py-2 pr-3 pl-8 text-[13px] outline-none focus:border-accent" />
                </label>
                <button type="submit" disabled={!repo.trim() || busy} className="rounded-lg bg-ink px-3.5 py-2 text-[13px] font-medium text-bg hover:opacity-85 disabled:opacity-40">
                  {busy ? "Loading…" : "Load"}
                </button>
              </form>
              <div className="flex flex-wrap gap-1.5">
                {SUGGESTED.map((name) => (
                  <button key={name} type="button" onClick={() => setRepo(name)} className="rounded-full border border-line px-2.5 py-0.5 font-mono text-[11.5px] text-muted hover:border-muted hover:text-ink">
                    {name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {tab === "config" && (
            <div className="space-y-3">
              <p className="text-[13px] text-muted">Paste a model&apos;s config.json from its Hugging Face page (Files tab).</p>
              <textarea value={pasted} onChange={(event) => setPasted(event.target.value)} spellCheck={false} placeholder='{ "model_type": "llama", "hidden_size": 4096, ... }' className="h-64 w-full rounded-lg border border-line bg-bg p-3 font-mono text-[12px] outline-none focus:border-accent" />
              <button
                type="button"
                disabled={!pasted.trim()}
                onClick={() =>
                  void attempt(() => {
                    const config = JSON.parse(pasted);
                    const name = String(config._name_or_path || config.architectures?.[0] || config.model_type || "Pasted config");
                    finish(fromHuggingFace(config, name), name);
                  })
                }
                className="rounded-lg bg-ink px-3.5 py-2 text-[13px] font-medium text-bg hover:opacity-85 disabled:opacity-40"
              >
                Load config
              </button>
            </div>
          )}

          {tab === "onnx" && (
            <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line px-6 py-12 text-center text-[13px] text-muted hover:border-muted">
              <FileUp className="size-7" />
              <span>
                Choose an <span className="text-ink">.onnx</span> file. It is read in this browser; nothing is uploaded.
              </span>
              <span className="text-[12px]">Export one from PyTorch with torch.onnx.export, or download one from the ONNX model zoo.</span>
              <input
                type="file"
                accept=".onnx"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void attempt(async () => finish(fromOnnx(new Uint8Array(await file.arrayBuffer()), file.name.replace(/\.onnx$/, "")), file.name));
                }}
              />
            </label>
          )}

          {tab === "blank" && (
            <div className="grid gap-3 sm:grid-cols-2">
              {(["vision", "language"] as const).map((kind) => (
                <button
                  key={kind}
                  type="button"
                  onClick={() => {
                    const { model } = parseText(BLANK[kind]);
                    finish(model, kind === "vision" ? "New image model" : "New language model");
                  }}
                  className="rounded-xl border border-line p-4 text-left hover:border-muted hover:bg-surface-2"
                >
                  <span className="text-[14px] font-medium">{kind === "vision" ? "Image model" : "Language model"}</span>
                  <span className="mt-1 block text-[12.5px] text-muted">{kind === "vision" ? "A small CNN with one skip connection, to rewire from." : "A tiny decoder: attention, feed-forward and residuals, ×4."}</span>
                  <pre className="mt-3 overflow-hidden rounded-lg bg-bg p-2.5 font-mono text-[10.5px] leading-relaxed text-muted">{BLANK[kind].split("\n").slice(1, 6).join("\n")}</pre>
                </button>
              ))}
            </div>
          )}

          {error && <p className="mt-4 rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</p>}
        </div>
      </div>
    </div>
  );
}
