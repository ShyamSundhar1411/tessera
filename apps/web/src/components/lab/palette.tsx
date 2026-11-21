"use client";

import { OPS } from "tessera-engine";
import { CATEGORY_COLOURS } from "@/lib/display";

const GROUPS: { label: string; ops: string[] }[] = [
  { label: "Image", ops: ["conv2d", "maxpool", "avgpool", "gap", "batchnorm"] },
  { label: "Dense", ops: ["linear", "flatten", "dropout", "layernorm"] },
  { label: "Join", ops: ["add", "concat", "mul"] },
  { label: "Activation", ops: ["relu", "gelu", "silu", "sigmoid", "softmax"] },
  { label: "Transformer", ops: ["embedding", "rmsnorm", "attention", "mla", "mlp", "moe", "lmhead"] },
];

/** Layers to drag onto the canvas while editing. */
export default function Palette() {
  return (
    <div className="scroll-quiet flex h-full w-36 flex-none flex-col gap-3 overflow-y-auto border-r border-line bg-surface p-2.5">
      <p className="text-[11.5px] leading-snug text-muted">Drag onto the canvas. A new layer reads from the selected one.</p>
      {GROUPS.map((group) => (
        <div key={group.label}>
          <p className="mb-1 text-[11px] text-muted">{group.label}</p>
          <div className="flex flex-col gap-1">
            {group.ops.map((op) => (
              <div
                key={op}
                draggable
                title={OPS[op]?.summary}
                onDragStart={(event) => {
                  event.dataTransfer.setData("application/x-tessera-op", op);
                  event.dataTransfer.effectAllowed = "copy";
                }}
                className="flex cursor-grab items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[11.5px] hover:border-muted active:cursor-grabbing"
              >
                <span className="size-2 rounded-full" style={{ background: CATEGORY_COLOURS[OPS[op]?.category ?? "reshape"] }} />
                {op}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
