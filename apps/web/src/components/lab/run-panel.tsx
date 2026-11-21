"use client";

import { ImagePlus, Network } from "lucide-react";
import { useEffect, useState } from "react";
import { catalogModel } from "tessera-engine";
import type { ClassifyResult, PretrainedEntry } from "tessera-lab";
import { useLab } from "@/lib/store";

/** Runs a real pretrained image classifier on a picture, in the browser. */
export default function RunPanel() {
  const load = useLab((state) => state.load);
  const [entries, setEntries] = useState<PretrainedEntry[]>([]);
  const [chosen, setChosen] = useState("");
  const [image, setImage] = useState<{ url: string; element: HTMLImageElement } | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void import("tessera-lab").then(({ PRETRAINED }) => {
      setEntries(PRETRAINED);
      setChosen((current) => current || PRETRAINED[0]?.id || "");
    });
  }, []);

  const entry = entries.find((item) => item.id === chosen);

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
    };
    element.src = url;
  };

  const run = async () => {
    if (!entry || !image) return;
    setBusy(true);
    setError(null);
    try {
      const { classify } = await import("tessera-lab");
      const output = await classify(entry, image.element, (progress) => {
        if (progress.stage === "download") setStatus(`Downloading ${entry.name} (${entry.sizeMB} MB)… ${progress.total ? Math.round(((progress.loaded ?? 0) / progress.total) * 100) : 0}%`);
        else setStatus(progress.stage === "load" ? "Starting the model…" : "Running…");
      });
      setResult(output);
      setStatus(`Ran on ${output.backend} in ${(output.seconds * 1000).toFixed(0)} ms.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4 p-4 text-[13px]">
      <p className="leading-relaxed text-muted">Real ImageNet classifiers, with their trained weights, running in this browser. Your picture never leaves the page.</p>
      <label className="block text-[12px] text-muted">
        Model
        <select value={chosen} onChange={(event) => setChosen(event.target.value)} className="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1 text-[12.5px] text-ink">
          {entries.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name} · {item.sizeMB} MB{item.precision === "uint8" ? " (quantised)" : ""}
            </option>
          ))}
        </select>
      </label>
      {entry?.catalogId && (
        <button type="button" onClick={() => load(0, catalogModel(entry.catalogId!)!, entry.name)} className="flex items-center gap-1.5 text-[12.5px] text-accent hover:underline">
          <Network className="size-3.5" />
          Show its architecture as A
        </button>
      )}

      <label
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          pick(event.dataTransfer.files[0]);
        }}
        className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line px-4 py-6 text-center text-[12.5px] text-muted hover:border-muted"
      >
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image.url} alt="Your picture" className="max-h-44 rounded-lg" />
        ) : (
          <>
            <ImagePlus className="size-6" />
            Drop a photo here, or click to choose one
          </>
        )}
        <input type="file" accept="image/*" hidden onChange={(event) => pick(event.target.files?.[0])} />
      </label>

      <button type="button" disabled={!image || !entry || busy} onClick={() => void run()} className="rounded-lg bg-ink px-3 py-1.5 text-[13px] font-medium text-bg hover:opacity-85 disabled:opacity-40">
        {busy ? "Working…" : "Classify"}
      </button>
      {status && <p className="font-mono text-[11.5px] text-muted">{status}</p>}
      {error && <p className="rounded-md border border-danger/50 bg-danger/10 px-2.5 py-2 text-[12.5px] text-danger">{error}</p>}

      {result && (
        <ul className="space-y-2">
          {result.top5.map((prediction) => (
            <li key={prediction.index}>
              <div className="flex justify-between gap-2 text-[12.5px]">
                <span className="truncate">{prediction.label}</span>
                <span className="font-mono text-muted">{(prediction.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-surface-3">
                <div className="h-full rounded-full bg-accent" style={{ width: `${prediction.probability * 100}%` }} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
