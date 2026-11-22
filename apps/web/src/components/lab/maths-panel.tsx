"use client";

import { useMemo } from "react";
import { formatBytes, formatCount, type Category } from "tessera-engine";
import { CATEGORY_COLOURS } from "@/lib/display";
import { computeByCategory, flopsPerToken, gradientPaths, receptiveField, scaling } from "@/lib/maths";
import { useLab } from "@/lib/store";
import Tex from "./tex";

// Worked out from the wiring alone, no training: how far gradients travel,
// what each output can see, where the compute goes, and what longer context
// costs. A and B side by side when both are loaded.

const COLOURS = ["#5fb4ff", "#f0569f"];
const CONTEXTS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

const shortN = (n: number) => (n >= 1048576 ? `${n / 1048576}M` : n >= 1024 ? `${n / 1024}K` : String(n));

function Section({ title, tex, children, note }: { title: string; tex?: string; children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <section className="space-y-2 border-b border-line px-4 py-4 last:border-b-0">
      <h3 className="text-[13px] font-semibold">{title}</h3>
      {tex && (
        <div className="rounded-md bg-surface-2 px-2 text-[13px]">
          <Tex tex={tex} display />
        </div>
      )}
      {children}
      {note && <p className="text-[11.5px] leading-relaxed text-muted">{note}</p>}
    </section>
  );
}

/** Log-log lines, one per model. */
function LogChart({ series, format, label }: { series: { name: string; colour: string; points: [number, number][] }[]; format: (v: number) => string; label: string }) {
  const width = 300;
  const height = 150;
  const pad = { l: 46, r: 8, t: 8, b: 20 };
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const lx = (v: number) => Math.log10(v);
  const [x0, x1] = [Math.min(...all.map((p) => lx(p[0]))), Math.max(...all.map((p) => lx(p[0])))];
  const [y0, y1] = [Math.min(...all.map((p) => lx(p[1]))), Math.max(...all.map((p) => lx(p[1])))];
  const x = (v: number) => pad.l + ((lx(v) - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (lx(v) - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
  const ticks = [y0, (y0 + y1) / 2, y1].map((v) => 10 ** v);
  return (
    <figure>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={label}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={width - pad.r} y1={y(v)} y2={y(v)} stroke="var(--line)" />
            <text x={pad.l - 4} y={y(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted)">
              {format(v)}
            </text>
          </g>
        ))}
        {[CONTEXTS[0], 4096, 131072, CONTEXTS[CONTEXTS.length - 1]].map((n) => (
          <text key={n} x={x(n)} y={height - 6} textAnchor="middle" fontSize="9" fill="var(--muted)">
            {shortN(n)}
          </text>
        ))}
        {series.map((s) => (
          <polyline key={s.name} fill="none" stroke={s.colour} strokeWidth="2" points={s.points.map(([px, py]) => `${x(px)},${y(py)}`).join(" ")} />
        ))}
      </svg>
      <figcaption className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span>{label}</span>
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1">
            <span className="h-0.5 w-3" style={{ background: s.colour }} />
            {s.name}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

export default function MathsPanel() {
  const slots = useLab((state) => state.slots);
  const models = useMemo(
    () =>
      slots
        .map((slot, index) => (slot ? { index, name: slot.model.name, model: slot.model, analysis: slot.analysis } : null))
        .filter((item): item is NonNullable<typeof item> => !!item && item.analysis.errors.length === 0)
        .map((item) => ({
          ...item,
          colour: COLOURS[item.index],
          paths: gradientPaths(item.model, item.analysis),
          field: receptiveField(item.model, item.analysis),
          compute: computeByCategory(item.model, item.analysis),
          scale: scaling(item.model, item.analysis),
        })),
    [slots],
  );
  if (!models.length) return <p className="p-4 text-[13px] text-muted">Load a model without errors to see its maths.</p>;
  const deepest = Math.max(...models.map((m) => m.paths.longest), 1);
  const scaled = models.filter((m) => m.scale);
  const fielded = models.filter((m) => m.field?.length);

  return (
    <div className="text-[12.5px]">
      <Section
        title="How far gradients travel"
        tex={String.raw`y = x + F(x) \;\Rightarrow\; \frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}`}
        note="Training sends the error backwards through every layer, and each layer it passes can shrink it. A skip connection adds an identity path, so the error can also reach early layers through only the shortest route. Counted in layers with weights (norms aside), every copy of a repeated block included."
      >
        {models.map((m) => (
          <div key={m.index} className="space-y-1">
            <p className="font-medium" style={{ color: m.colour }}>
              {m.name}
            </p>
            {(["longest", "shortest"] as const).map((key) => (
              <div key={key} className="flex items-center gap-2">
                <span className="w-16 text-[11.5px] text-muted">{key}</span>
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
                  <span className="block h-full rounded-full" style={{ width: `${Math.max(1.5, (m.paths[key] / deepest) * 100)}%`, background: m.colour, opacity: key === "shortest" ? 0.6 : 1 }} />
                </span>
                <span className="w-10 text-right font-mono text-[11.5px]">{m.paths[key]}</span>
              </div>
            ))}
            <p className="text-[11.5px] text-muted">
              {m.paths.skips ? `${m.paths.skips} residual additions` : "No residual additions"}
              {m.paths.joins ? `, ${m.paths.joins} concatenations` : ""}.{" "}
              {m.paths.shortest < m.paths.longest ? `The error can skip ${m.paths.longest - m.paths.shortest} of ${m.paths.longest} layers.` : "Every gradient passes through every layer, so depth is hard to train."}
            </p>
          </div>
        ))}
      </Section>

      {fielded.length > 0 && (
        <Section
          title="What each output can see"
          tex={String.raw`r_l = r_{l-1} + (k_l - 1)\prod_{i<l} s_i`}
          note="The receptive field: how many input pixels, across, can influence one value of a layer. Kernels widen it a little; strides multiply how fast it grows."
        >
          {fielded.map((m) => {
            const points = m.field!;
            const top = Math.max(...fielded.flatMap((f) => f.field!.map((p) => p.field)));
            const final = points[points.length - 1];
            return (
              <div key={m.index}>
                <p className="font-medium" style={{ color: m.colour }}>
                  {m.name}
                </p>
                <div className="mt-1 flex h-16 items-end gap-px">
                  {points.map((p, i) => (
                    <div key={i} className="flex-1 rounded-t-[1px]" title={`${p.label}: ${p.field}px, ${p.size}×${p.size} map`} style={{ height: `${(p.field / top) * 100}%`, background: m.colour, opacity: 0.35 + 0.65 * (i / points.length) }} />
                  ))}
                </div>
                <p className="mt-1 text-[11.5px] text-muted">
                  After the last spatial layer each value sees {final.field}×{final.field} pixels, and neighbouring values are {final.jump} pixels apart.
                </p>
              </div>
            );
          })}
        </Section>
      )}

      <Section title="Where the compute goes" note="Floating point operations for one forward pass, by kind of layer.">
        {models.map((m) => {
          const total = m.compute.reduce((sum, [, v]) => sum + v, 0) || 1;
          return (
            <div key={m.index}>
              <p className="flex justify-between font-medium">
                <span style={{ color: m.colour }}>{m.name}</span>
                <span className="font-mono text-[11.5px] text-muted">{formatCount(total)}FLOPs</span>
              </p>
              <div className="mt-1 flex h-3 overflow-hidden rounded-full">
                {m.compute.map(([category, value]) => (
                  <span key={category} title={`${category}: ${((value / total) * 100).toFixed(1)}%`} style={{ width: `${(value / total) * 100}%`, background: CATEGORY_COLOURS[category as Category] }} />
                ))}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-muted">
                {m.compute.slice(0, 4).map(([category, value]) => (
                  <span key={category} className="flex items-center gap-1">
                    <span className="size-2 rounded-full" style={{ background: CATEGORY_COLOURS[category as Category] }} />
                    {category} {((value / total) * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </Section>

      {scaled.length > 0 && (
        <Section
          title="What longer context costs"
          tex={String.raw`\text{FLOPs per token}(n) \approx 2N_{\text{active}} + 4\,n\sum_{\ell} h_\ell\, d_{k}`}
          note={
            <>
              The first term is the weights, the same for every token. The second is attention comparing the token with all n before it, so it grows with context.{" "}
              {scaled.map((m) => {
                const s = m.scale!;
                const cross = s.linear / s.quadratic;
                return (
                  <span key={m.index} className="block pt-1">
                    <span style={{ color: m.colour }}>{m.name}</span>: attention overtakes the weights at about {shortN(Math.round(cross))} tokens{s.window ? `, but a ${shortN(s.window)} sliding window caps it` : ""}. Key/value cache: {formatBytes(s.kvBytes)} per token, {formatBytes(s.kvBytes * 131072)} at 128K.
                  </span>
                );
              })}
            </>
          }
        >
          <LogChart label="FLOPs per token vs context" format={(v) => `${formatCount(v)}`} series={scaled.map((m) => ({ name: m.name, colour: m.colour, points: CONTEXTS.map((n) => [n, flopsPerToken(m.scale!, n)]) }))} />
          <LogChart label="Key/value cache vs context" format={(v) => formatBytes(v)} series={scaled.map((m) => ({ name: m.name, colour: m.colour, points: CONTEXTS.map((n) => [n, m.scale!.kvBytes * n]) }))} />
        </Section>
      )}
    </div>
  );
}
