"use client";

import { formatBytes, formatCount, type ArgValue, type ChangeKind } from "tessera-engine";
import { useLab } from "@/lib/store";
import { cn } from "@/lib/utils";

const BADGE: Record<ChangeKind, string> = {
  same: "text-muted",
  changed: "bg-changed/15 text-changed",
  added: "bg-added/15 text-added",
  removed: "bg-removed/15 text-removed",
};

const show = (value: ArgValue | undefined) => (value === undefined ? "-" : value === true ? "yes" : value === false ? "no" : String(value));

export default function ComparePanel() {
  const slots = useLab((state) => state.slots);
  const result = useLab((state) => state.diff);
  const select = useLab((state) => state.select);
  const setExpanded = useLab((state) => state.setExpanded);
  const [a, b] = slots;
  if (!a || !b || !result)
    return (
      <div className="p-4 text-[13px] leading-relaxed text-muted">
        <p className="font-medium text-ink">Compare two architectures</p>
        <p className="mt-1">Load a second model with “Add B” in the toolbar: headline numbers, differing settings and changed blocks appear here, and both drawings mark what changed.</p>
      </div>
    );
  const [ma, mb] = result.metrics;
  const rows: { label: string; a: number | null; b: number | null; format: (n: number) => string; lowerIsBetter?: boolean }[] = [
    { label: "Parameters", a: ma.params, b: mb.params, format: formatCount },
    { label: "Active per token", a: ma.activeParams, b: mb.activeParams, format: formatCount },
    { label: a.model.kind === "language" ? "Compute per token" : "Compute per image", a: ma.flops, b: mb.flops, format: (n) => `${formatCount(n)}FLOPs` },
    { label: "Weights in 16-bit", a: ma.weightBytes, b: mb.weightBytes, format: formatBytes },
    { label: "KV cache per token", a: ma.kvBytesPerToken, b: mb.kvBytesPerToken, format: formatBytes },
    { label: "Layers with weights", a: ma.weightLayers, b: mb.weightLayers, format: (n) => String(n) },
  ];
  const differing = result.settings.filter((row) => !row.same);
  const changes = result.changes.filter((change) => change.kind !== "same");

  return (
    <div className="space-y-5 p-4 text-[13px]">
      <table className="w-full text-[12.5px]">
        <thead className="text-[11.5px] text-muted">
          <tr>
            <th className="pb-1.5 text-left font-normal" />
            <th className="pb-1.5 text-right font-normal">A</th>
            <th className="pb-1.5 text-right font-normal">B</th>
            <th className="pb-1.5 text-right font-normal">B vs A</th>
          </tr>
        </thead>
        <tbody>
          {rows
            .filter((row) => row.a !== null || row.b !== null)
            .map((row) => {
              const ratio = row.a && row.b ? row.b / row.a : null;
              return (
                <tr key={row.label} className="border-t border-line">
                  <td className="py-1.5 text-muted">{row.label}</td>
                  <td className="py-1.5 text-right font-mono">{row.a === null ? "-" : row.format(row.a)}</td>
                  <td className="py-1.5 text-right font-mono">{row.b === null ? "-" : row.format(row.b)}</td>
                  <td className={cn("py-1.5 text-right font-mono", ratio && ratio > 1.001 ? "text-changed" : ratio && ratio < 0.999 ? "text-added" : "text-muted")}>
                    {ratio === null ? "" : `${ratio >= 1 ? "+" : ""}${((ratio - 1) * 100).toFixed(ratio > 10 ? 0 : 1)}%`}
                  </td>
                </tr>
              );
            })}
        </tbody>
      </table>

      <section>
        <h3 className="text-[13px] font-medium">Settings that differ</h3>
        {differing.length === 0 ? (
          <p className="mt-1 text-[12.5px] text-muted">Every headline setting matches.</p>
        ) : (
          <table className="mt-1.5 w-full text-[12px]">
            <tbody>
              {differing.map((row) => (
                <tr key={row.key} className="border-t border-line align-top">
                  <td className="w-28 py-1.5 pr-2 text-muted">{row.key}</td>
                  <td className="py-1.5 pr-2 font-mono">{show(row.a)}</td>
                  <td className="py-1.5 font-mono text-accent">{show(row.b)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <h3 className="text-[13px] font-medium">Blocks</h3>
        <p className="mt-0.5 text-[12px] text-muted">
          {changes.length === 0 ? "Same blocks, wired the same way." : `${changes.length} of ${result.changes.length} differ. Click one to find it on the drawings.`}
        </p>
        <ul className="mt-2 space-y-1.5">
          {changes.map((change, index) => (
            <li key={index}>
              <button
                type="button"
                onClick={() => {
                  if (change.a) {
                    if (change.a.id !== change.a.nodeIds[0]) setExpanded(0, [...new Set([...a.expanded, change.a.id])]);
                    select({ slot: 0, id: change.a.id });
                  }
                  if (change.b && change.b.id !== change.b.nodeIds[0]) setExpanded(1, [...new Set([...b.expanded, change.b.id])]);
                }}
                className="w-full rounded-lg border border-line px-2.5 py-2 text-left hover:border-muted"
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="truncate text-[12.5px] font-medium">{change.b?.label ?? change.a?.label}</span>
                  <span className={cn("rounded px-1.5 py-0.5 text-[11px]", BADGE[change.kind])}>{change.kind}</span>
                </span>
                {change.detail && <span className="mt-0.5 block font-mono text-[11px] text-muted">{change.detail}</span>}
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
