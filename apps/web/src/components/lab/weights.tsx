"use client";

import { Download } from "lucide-react";
import type { Model } from "tessera-engine";
import { useMemo, useState } from "react";
import { VISION_SIZES, isGpt2, loadRealGpt2, loadRealVision, visionWeightsId } from "@/lib/trained";
import { cn } from "@/lib/utils";

const LABEL = { random: "random weights", trained: "trained here", pretrained: "real published weights" };

/** Which weights a view is running. */
export function WeightsBadge({ weights }: { weights: "random" | "trained" | "pretrained" }) {
  return <span className={cn("rounded-full px-2 py-0.5 text-[11px] whitespace-nowrap", weights === "random" ? "bg-warn/15 text-warn" : "bg-added/15 text-added")}>{LABEL[weights]}</span>;
}

/** Offers real published weights when the slot holds a model that has them (GPT-2 small, torchvision ResNets). */
export function RealWeightsButton({ slot, model, weights }: { slot: 0 | 1; model: Model; weights: "random" | "trained" | "pretrained" }) {
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const gpt2 = isGpt2(model);
  const vision = useMemo(() => (gpt2 ? undefined : visionWeightsId(model)), [model, gpt2]);
  if ((!gpt2 && !vision) || weights === "pretrained") return null;
  const megabytes = gpt2 ? 548 : VISION_SIZES[vision!];
  const start = async () => {
    setError(null);
    setStatus("Starting…");
    const downloading = (loaded: number, total: number) => setStatus(`Downloading ${Math.round(loaded / 1e6)} of ${Math.round(total / 1e6)} MB`);
    try {
      if (gpt2) await loadRealGpt2(slot, model, (progress) => (progress.stage === "download" ? downloading(progress.loaded, progress.total) : setStatus("Putting the weights on the GPU…")));
      else await loadRealVision(slot, model, vision!, downloading);
      setStatus(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus(null);
    }
  };
  return (
    <span className="flex items-center gap-2">
      <button type="button" disabled={!!status} onClick={start} className="flex items-center gap-1.5 rounded-lg bg-accent px-2.5 py-1 text-[12px] font-medium text-accent-ink disabled:opacity-70">
        <Download className="size-3.5" />
        {status ?? `Use ${gpt2 ? "OpenAI's" : "torchvision's"} real weights (${megabytes} MB, once)`}
      </button>
      {error && <span className="text-[11.5px] text-danger">{error}</span>}
    </span>
  );
}
