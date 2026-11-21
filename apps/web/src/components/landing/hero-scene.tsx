"use client";

import dynamic from "next/dynamic";

// The lab's own 3D view of the starter model: feature maps, kernel cones and neurons, with the forward pass running.
const Graph3D = dynamic(() => import("../lab/graph-3d"), { ssr: false, loading: () => <div className="h-full w-full animate-pulse rounded-2xl bg-surface" /> });

export default function HeroScene() {
  return (
    <div className="relative h-[340px] overflow-hidden rounded-2xl border border-line sm:h-[420px]">
      <Graph3D slot={0} compact />
      <p className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-surface/85 px-2.5 py-1 text-[12px] text-muted backdrop-blur">A small CNN, live. Drag to turn it.</p>
    </div>
  );
}
