"use client";

import { useEffect, useState } from "react";
import { shotUrl } from "@/lib/shots";
import { cn } from "@/lib/utils";
import HeroScene from "./hero-scene";

// The hero: everything the lab does, one tab at a time. The first tab is the
// live 3D view; the rest are screenshots of the lab running, real GPT-2 and
// ResNet weights included. It moves on by itself until someone picks a tab.

export const SHOTS = [
  { id: "3d", label: "3D", caption: "The tensors themselves: feature maps, kernel cones and neurons, with the forward pass running. Drag to turn it." },
  { id: "architecture", label: "Architecture", caption: "Every layer drawn as what it does. The original Transformer beside ResNet-18, changed blocks marked." },
  { id: "trace", label: "Trace", caption: "A drawn digit followed through a CNN: each layer's equation with real sizes, and its real feature maps painted onto the 3D." },
  { id: "lens", label: "Logit lens", caption: "OpenAI's real GPT-2 after \"To be\": what it would guess at every layer, until it settles on \"able\"." },
  { id: "working", label: "Every number", caption: "Click any number to see the arithmetic that made it, with the model's actual inputs and weights." },
  { id: "run", label: "Run GPT-2", caption: "GPT-2's published weights writing in your browser. Hover a token to see the five it weighed up." },
  { id: "maths", label: "Maths", caption: "From the wiring alone: gradient paths, receptive fields, compute, and what longer context costs." },
  { id: "compare", label: "Compare", caption: "Mistral 7B against Mixtral 8x7B: parameters, active parameters, memory, cache and every setting that differs." },
  { id: "papers", label: "Papers", caption: "23 landmark papers from 1998 to 2025, each a model you can open, trace and take apart." },
] as const;

export default function Showcase() {
  const [index, setIndex] = useState(0);
  const [held, setHeld] = useState(false);
  const [chosen, setChosen] = useState(false);

  useEffect(() => {
    if (held || chosen) return;
    const timer = window.setTimeout(() => setIndex((i) => (i + 1) % SHOTS.length), index === 0 ? 7000 : 4800);
    return () => window.clearTimeout(timer);
  }, [index, held, chosen]);

  const shot = SHOTS[index];
  return (
    <div onPointerEnter={() => setHeld(true)} onPointerLeave={() => setHeld(false)}>
      <div className="scroll-quiet -mx-1 mb-2 flex gap-1 overflow-x-auto px-1 pb-1" role="tablist" aria-label="What the lab does">
        {SHOTS.map((item, i) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={i === index}
            onClick={() => {
              setIndex(i);
              setChosen(true);
            }}
            className={cn("relative flex-none overflow-hidden rounded-full border px-3 py-1 text-[12.5px] whitespace-nowrap transition-colors", i === index ? "border-accent bg-accent/10 text-ink" : "border-line text-muted hover:text-ink")}
          >
            {item.label}
            {/* How long until the next tab, while it is moving on by itself. */}
            {i === index && !held && !chosen && <span key={index} className="demo-progress absolute inset-x-0 bottom-0 h-0.5 origin-left bg-accent" style={{ animationDuration: `${index === 0 ? 7000 : 4800}ms` }} />}
          </button>
        ))}
      </div>
      <div className="relative aspect-[16/10] overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl">
        {shot.id === "3d" ? (
          <div className="absolute inset-0">
            <HeroScene fill />
          </div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img key={shot.id} src={shotUrl(shot.id)} alt={shot.caption} className="rise absolute inset-0 size-full object-cover object-top" style={{ animationDuration: "0.5s" }} />
        )}
      </div>
      <p className="mt-2.5 min-h-[2.6em] text-[13px] leading-snug text-muted">{shot.caption}</p>
    </div>
  );
}
