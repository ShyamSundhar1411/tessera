"use client";

import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";

// The lab's own 3D view of the starter model: feature maps, kernel cones and neurons, with the forward pass running.
const Graph3D = dynamic(() => import("../lab/graph-3d"), { ssr: false, loading: () => <div className="h-full w-full animate-pulse bg-surface" /> });

/** `fill` makes it take its parent's size, without its own frame and caption. */
export default function HeroScene({ fill = false }: { fill?: boolean }) {
  return (
    <div className={cn("relative overflow-hidden", fill ? "h-full" : "h-[340px] rounded-2xl border border-line sm:h-[420px]")}>
      <Graph3D slot={0} compact />
      {!fill && <p className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-surface/85 px-2.5 py-1 text-[12px] text-muted backdrop-blur">A small CNN, live. Drag to turn it.</p>}
    </div>
  );
}
