"use client";

import { Check, Copy, Download, ExternalLink } from "lucide-react";
import { useMemo, useState } from "react";
import { toNotebook, toPyTorch, toText } from "tessera-engine";
import { useLab } from "@/lib/store";
import { cn, download } from "@/lib/utils";

type Format = "pytorch" | "text" | "json" | "notebook";
const FORMATS: { id: Format; label: string }[] = [
  { id: "pytorch", label: "PyTorch" },
  { id: "notebook", label: "Colab" },
  { id: "text", label: "Text" },
  { id: "json", label: "JSON" },
];

const slug = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "model";

export default function CodePanel() {
  const slots = useLab((state) => state.slots);
  const active = useLab((state) => state.active);
  const [format, setFormat] = useState<Format>("pytorch");
  const [copied, setCopied] = useState(false);
  const [epochs, setEpochs] = useState(5);
  const [seeds, setSeeds] = useState(3);
  const slot = slots[active];
  const both = slots.filter(Boolean).map((item) => item!.model);

  const code = useMemo(() => {
    if (!slot) return "";
    if (format === "pytorch") return toPyTorch(slot.model);
    if (format === "text") return toText(slot.model);
    if (format === "json") return JSON.stringify({ ...slot.model }, null, 2);
    return "";
  }, [slot, format]);

  if (!slot) return null;
  const notebook = () => toNotebook(both, { epochs, steps: epochs * 300, seeds: Array.from({ length: seeds }, (_, i) => i) });
  const name = slug(slot.model.name);

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-none gap-1 border-b border-line p-2">
        {FORMATS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setFormat(item.id)}
            aria-pressed={format === item.id}
            className={cn("rounded-md px-2.5 py-1 text-[12.5px] text-muted hover:text-ink", format === item.id && "bg-surface-3 text-ink")}
          >
            {item.label}
          </button>
        ))}
      </div>
      {format === "notebook" ? (
        <div className="space-y-4 p-4 text-[13px] leading-relaxed">
          <p>
            A notebook that trains {both.length > 1 ? "A and B the same way" : "this model"} on Google Colab&apos;s free GPU and reports loss, accuracy, error rate, top-5,
            per-class precision, recall and F1, confusion matrices and how much each varies across seeds, with a test of whether the gap is bigger than the noise.
          </p>
          {both.every((model) => model.kind === "language") && (
            <p className="text-muted">Language models are shrunk to about 30M parameters with the same wiring (attention type, head sharing, experts, norms) and trained on TinyStories.</p>
          )}
          <div className="grid grid-cols-2 gap-3">
            <label className="text-[12px] text-muted">
              {both[0]?.kind === "language" ? "Training rounds (x300 steps)" : "Epochs"}
              <input type="number" min={1} max={100} value={epochs} onChange={(event) => setEpochs(Math.max(1, Number(event.target.value) || 1))} className="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-ink" />
            </label>
            <label className="text-[12px] text-muted">
              Seeds
              <input type="number" min={1} max={10} value={seeds} onChange={(event) => setSeeds(Math.max(1, Number(event.target.value) || 1))} className="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-ink" />
            </label>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => download(notebook(), `${both.map((model) => slug(model.name)).join("-vs-")}.ipynb`, "application/x-ipynb+json")} className="flex items-center gap-1.5 rounded-lg bg-ink px-3 py-1.5 text-[13px] font-medium text-bg hover:opacity-85">
              <Download className="size-4" />
              Download notebook
            </button>
            <a href="https://colab.research.google.com/#create=true" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] hover:border-muted">
              Open Colab
              <ExternalLink className="size-3.5" />
            </a>
          </div>
          <p className="text-[12px] text-muted">In Colab: File → Upload notebook, choose the file, then Runtime → Change runtime type → T4 GPU, and Run all.</p>
        </div>
      ) : (
        <>
          <div className="flex flex-none items-center justify-end gap-1.5 px-3 pt-2">
            <button
              type="button"
              onClick={async () => {
                await navigator.clipboard.writeText(code);
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
              className="flex items-center gap-1 rounded-md border border-line px-2 py-1 text-[12px] hover:border-muted"
            >
              {copied ? <Check className="size-3.5 text-added" /> : <Copy className="size-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
            <button type="button" onClick={() => download(code, `${name}.${format === "pytorch" ? "py" : format === "json" ? "json" : "txt"}`)} className="flex items-center gap-1 rounded-md border border-line px-2 py-1 text-[12px] hover:border-muted">
              <Download className="size-3.5" />
              Download
            </button>
          </div>
          <pre className="scroll-quiet min-h-0 flex-1 overflow-auto p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre">{code}</pre>
        </>
      )}
    </div>
  );
}
