"use client";

import { Download } from "lucide-react";
import { useState } from "react";
import type { Model } from "tessera-engine";
import { isGpt2, loadRealGpt2 } from "@/lib/trained";
import { cn } from "@/lib/utils";

const LABEL = { random: "random weights", trained: "trained here", pretrained: "real published weights" };

/** Which weights a view is running. */
export function WeightsBadge({ weights }: { weights: "random" | "trained" | "pretrained" }) {
  return <span className={cn("rounded-full px-2 py-0.5 text-[11px] whitespace-nowrap", weights === "random" ? "bg-warn/15 text-warn" : "bg-added/15 text-added")}>{LABEL[weights]}</span>;
}

/** Offers OpenAI's weights when the slot holds GPT-2 small. */
export function RealWeightsButton({ slot, model, weights }: { slot: 0 | 1; model: Model; weights: "random" | "trained" | "pretrained" }) {
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (!isGpt2(model) || weights === "pretrained") return null;
  const start = async () => {
    setError(null);
    setStatus("Starting…");
    try {
      await loadRealGpt2(slot, model, (progress) =>
        setStatus(progress.stage === "download" ? `Downloading ${Math.round(progress.loaded / 1e6)} of ${Math.round(progress.total / 1e6)} MB` : "Putting the weights on the GPU…"),
      );
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
        {status ?? "Use OpenAI's real weights (548 MB, once)"}
      </button>
      {error && <span className="text-[11.5px] text-danger">{error}</span>}
    </span>
  );
}
