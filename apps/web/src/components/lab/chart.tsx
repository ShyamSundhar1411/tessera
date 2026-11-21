"use client";

/** A small line chart with a shaded band for spread across runs. */
export function BandChart({ series, height = 120, label, percent }: { series: { name: string; colour: string; mean: number[]; std: number[] }[]; height?: number; label: string; percent?: boolean }) {
  const width = 300;
  const pad = { l: 34, r: 8, t: 8, b: 18 };
  const points = series.flatMap((s) => s.mean.map((m, i) => [m - (s.std[i] ?? 0), m + (s.std[i] ?? 0)]).flat());
  if (!points.length) return <p className="py-6 text-center text-[12px] text-muted">{label}: waiting for the first epoch</p>;
  let lo = Math.min(...points);
  let hi = Math.max(...points);
  if (hi - lo < 1e-6) {
    lo -= 0.5;
    hi += 0.5;
  }
  const epochs = Math.max(...series.map((s) => s.mean.length));
  const x = (i: number) => pad.l + (epochs <= 1 ? (width - pad.l - pad.r) / 2 : (i / (epochs - 1)) * (width - pad.l - pad.r));
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const fmt = (v: number) => (percent ? `${(v * 100).toFixed(0)}%` : v.toFixed(2));
  return (
    <figure>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={label}>
        {[lo, (lo + hi) / 2, hi].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={width - pad.r} y1={y(v)} y2={y(v)} stroke="var(--line)" />
            <text x={pad.l - 4} y={y(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted)">
              {fmt(v)}
            </text>
          </g>
        ))}
        {series.map((s) => {
          const upper = s.mean.map((m, i) => `${x(i)},${y(m + (s.std[i] ?? 0))}`);
          const lower = s.mean.map((m, i) => `${x(i)},${y(m - (s.std[i] ?? 0))}`).reverse();
          return (
            <g key={s.name}>
              {s.mean.length > 1 && <polygon points={[...upper, ...lower].join(" ")} fill={s.colour} opacity="0.18" />}
              <polyline points={s.mean.map((m, i) => `${x(i)},${y(m)}`).join(" ")} fill="none" stroke={s.colour} strokeWidth="2" />
              {s.mean.map((m, i) => (
                <circle key={i} cx={x(i)} cy={y(m)} r="2.2" fill={s.colour} />
              ))}
            </g>
          );
        })}
        <text x={width - pad.r} y={height - 4} textAnchor="end" fontSize="9" fill="var(--muted)">
          epoch {epochs}
        </text>
      </svg>
      <figcaption className="text-[11.5px] text-muted">{label}</figcaption>
    </figure>
  );
}

/** A confusion matrix: rows are the true class, columns the predicted class. */
export function Confusion({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse font-mono text-[9.5px]">
        <thead>
          <tr>
            <th className="p-0.5 text-right font-normal text-muted">true ↓ / predicted →</th>
            {classes.map((name) => (
              <th key={name} className="h-14 p-0.5 align-bottom font-normal text-muted">
                <span className="inline-block w-4 origin-bottom-left -rotate-60 whitespace-nowrap">{name.slice(0, 8)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="pr-1 text-right font-normal whitespace-nowrap text-muted">{classes[i]?.slice(0, 10)}</th>
              {row.map((value, j) => (
                <td key={j} className="size-5 text-center" style={{ background: `color-mix(in srgb, ${i === j ? "var(--added)" : "var(--accent)"} ${Math.round((value / max) * 85)}%, transparent)` }} title={`${classes[i]} predicted as ${classes[j]}: ${value}`}>
                  {value > 0 ? value : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
