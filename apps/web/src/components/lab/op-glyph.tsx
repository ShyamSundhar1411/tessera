"use client";

import type { Shape } from "tessera-engine";

// A small drawing of what each operation does, so a layer reads at a glance:
// stacked feature maps under a kernel for a convolution, a fully connected fan
// for a linear layer, a causal attention matrix, the real activation curve, a
// bell curve for a norm, a join symbol for a skip. Drawn in the layer's colour.

const S = 40;

/** A function's curve across x in [-3, 3], scaled into the glyph. */
function curve(f: (x: number) => number, lo: number, hi: number) {
  const points: string[] = [];
  for (let i = 0; i <= 24; i++) {
    const x = -3 + (i / 24) * 6;
    const y = (f(x) - lo) / (hi - lo);
    points.push(`${6 + (i / 24) * 28},${34 - y * 28}`);
  }
  return points.join(" ");
}

function Plot({ f, lo, hi, colour }: { f: (x: number) => number; lo: number; hi: number; colour: string }) {
  const zeroY = 34 - ((0 - lo) / (hi - lo)) * 28;
  return (
    <>
      <line x1="5" x2="35" y1={zeroY} y2={zeroY} stroke="currentColor" strokeOpacity="0.25" />
      <line x1="20" x2="20" y1="5" y2="35" stroke="currentColor" strokeOpacity="0.25" />
      <polyline points={curve(f, lo, hi)} fill="none" stroke={colour} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </>
  );
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

export function OpGlyph({ op, shape, colour, causal = true }: { op?: string; shape: Shape; colour: string; causal?: boolean }) {
  const soft = { fill: colour, fillOpacity: 0.18, stroke: colour, strokeWidth: 1.2 };
  let body: React.ReactNode;
  switch (op) {
    case "input":
      body =
        shape.length === 3 ? (
          <>
            <rect x="7" y="9" width="26" height="22" rx="3" {...soft} />
            <path d="M9 28 L17 19 L22 24 L26 20 L31 28 Z" fill={colour} fillOpacity="0.7" />
            <circle cx="27" cy="14" r="2.2" fill={colour} />
          </>
        ) : (
          [0, 1, 2, 3, 4].map((i) => <rect key={i} x={5 + i * 6.2} y="16" width="5" height="8" rx="1" fill={colour} fillOpacity={0.35 + i * 0.12} />)
        );
      break;
    case "conv2d":
    case "upconv2d":
      body = (
        <>
          {[0, 1, 2].map((i) => (
            <rect key={i} x={6 + i * 4} y={14 - i * 4} width="18" height="18" rx="2" fill={colour} fillOpacity={0.12 + i * 0.12} stroke={colour} strokeWidth="1" />
          ))}
          <rect x="18" y="10" width="6" height="6" fill={colour} />
          <path d="M24 10 L34 22 M24 16 L34 22" stroke={colour} strokeWidth="1" />
          <circle cx="34" cy="22" r="1.8" fill={colour} />
        </>
      );
      break;
    case "maxpool":
    case "avgpool":
      body = (
        <>
          {Array.from({ length: 16 }, (_, i) => (
            <rect key={i} x={5 + (i % 4) * 5} y={10 + Math.floor(i / 4) * 5} width="4.2" height="4.2" fill={colour} fillOpacity={i % 4 < 2 && i < 8 ? 0.8 : 0.22} />
          ))}
          <path d="M26 20 L29 20" stroke={colour} strokeWidth="1.4" />
          {[0, 1, 2, 3].map((i) => (
            <rect key={i} x={30 + (i % 2) * 4.2} y={16 + Math.floor(i / 2) * 4.2} width="3.6" height="3.6" fill={colour} fillOpacity={i === 0 ? 0.9 : 0.3} />
          ))}
        </>
      );
      break;
    case "gap":
    case "mean_pool":
      body = (
        <>
          {Array.from({ length: 9 }, (_, i) => (
            <rect key={i} x={5 + (i % 3) * 6} y={11 + Math.floor(i / 3) * 6} width="5" height="5" fill={colour} fillOpacity="0.35" />
          ))}
          <path d="M24 20 L30 20" stroke={colour} strokeWidth="1.4" />
          <circle cx="33" cy="20" r="3.2" fill={colour} />
        </>
      );
      break;
    case "flatten":
    case "to_sequence":
      body = (
        <>
          {Array.from({ length: 4 }, (_, i) => (
            <rect key={i} x={6 + (i % 2) * 6} y={13 + Math.floor(i / 2) * 6} width="5" height="5" fill={colour} fillOpacity="0.45" />
          ))}
          <path d="M20 20 L24 20" stroke={colour} strokeWidth="1.4" />
          {[0, 1, 2, 3].map((i) => (
            <rect key={i} x="27" y={9 + i * 6} width="6" height="5" fill={colour} fillOpacity="0.7" />
          ))}
        </>
      );
      break;
    case "linear":
    case "lmhead": {
      const left = [11, 20, 29];
      const right = op === "lmhead" ? [8, 14, 20, 26, 32] : [11, 20, 29];
      body = (
        <>
          {left.flatMap((a) => right.map((b) => <line key={`${a}-${b}`} x1="9" y1={a} x2="31" y2={b} stroke={colour} strokeOpacity="0.35" strokeWidth="0.8" />))}
          {left.map((y) => (
            <circle key={`l${y}`} cx="9" cy={y} r="2.6" fill={colour} />
          ))}
          {right.map((y) => (
            <circle key={`r${y}`} cx="31" cy={y} r={op === "lmhead" ? 1.9 : 2.6} fill={colour} />
          ))}
        </>
      );
      break;
    }
    case "batchnorm":
    case "layernorm":
    case "rmsnorm":
      body = (
        <>
          <Plot f={(x) => Math.exp(-x * x / 1.4)} lo={-0.1} hi={1.05} colour={colour} />
        </>
      );
      break;
    case "relu":
    case "relu6":
      body = <Plot f={(x) => (op === "relu6" ? Math.min(Math.max(0, x), 2) : Math.max(0, x))} lo={-1} hi={3} colour={colour} />;
      break;
    case "gelu":
    case "hardswish":
      body = <Plot f={gelu} lo={-1} hi={3} colour={colour} />;
      break;
    case "silu":
      body = <Plot f={(x) => x / (1 + Math.exp(-x))} lo={-1} hi={3} colour={colour} />;
      break;
    case "sigmoid":
      body = <Plot f={(x) => 1 / (1 + Math.exp(-2 * x))} lo={-0.1} hi={1.1} colour={colour} />;
      break;
    case "tanh":
      body = <Plot f={(x) => Math.tanh(x)} lo={-1.2} hi={1.2} colour={colour} />;
      break;
    case "softmax":
      body = [3, 7, 18, 9, 4].map((h, i) => <rect key={i} x={7 + i * 5.6} y={33 - h * 1.3} width="4.4" height={h * 1.3} rx="1" fill={colour} fillOpacity={h === 18 ? 1 : 0.45} />);
      break;
    case "dropout":
      body = Array.from({ length: 16 }, (_, i) => (
        <circle key={i} cx={8 + (i % 4) * 8} cy={8 + Math.floor(i / 4) * 8} r="2.6" fill={[1, 6, 11, 12].includes(i) ? "none" : colour} stroke={colour} strokeWidth="1" />
      ));
      break;
    case "add":
    case "mul":
      body = (
        <>
          <circle cx="20" cy="20" r="11" {...soft} strokeWidth="1.6" />
          <path d={op === "add" ? "M20 14 V26 M14 20 H26" : "M16 16 L24 24 M24 16 L16 24"} stroke={colour} strokeWidth="2.2" strokeLinecap="round" />
        </>
      );
      break;
    case "concat":
      body = (
        <>
          <rect x="6" y="10" width="7" height="20" rx="1.5" fill={colour} fillOpacity="0.4" />
          <rect x="15" y="10" width="7" height="20" rx="1.5" fill={colour} fillOpacity="0.7" />
          <path d="M24 20 L28 20" stroke={colour} strokeWidth="1.4" />
          <rect x="29" y="10" width="6" height="20" rx="1.5" fill={colour} />
        </>
      );
      break;
    case "embedding":
      body = (
        <>
          {[0, 1, 2, 3, 4].map((i) => (
            <rect key={i} x="8" y={7 + i * 5.4} width="24" height="4.2" rx="1" fill={colour} fillOpacity={i === 2 ? 1 : 0.22} />
          ))}
          <path d="M3 18.7 L7 18.7" stroke={colour} strokeWidth="1.6" />
        </>
      );
      break;
    case "posembed":
      body = (
        <>
          <polyline points={curve((x) => Math.sin(x * 2.2), -1.2, 1.2)} fill="none" stroke={colour} strokeWidth="2" />
          <polyline points={curve((x) => Math.cos(x * 1.1), -1.2, 1.2)} fill="none" stroke={colour} strokeOpacity="0.45" strokeWidth="1.4" />
        </>
      );
      break;
    case "attention":
    case "cross_attention":
    case "mla": {
      const n = 5;
      body = Array.from({ length: n * n }, (_, i) => {
        const [r, c] = [Math.floor(i / n), i % n];
        const hidden = causal && op !== "cross_attention" && c > r;
        const weight = hidden ? 0 : 0.25 + 0.75 * (((r * 7 + c * 3) % 5) / 4);
        return <rect key={i} x={7 + c * 5.4} y={7 + r * 5.4} width="4.6" height="4.6" rx="0.8" fill={colour} fillOpacity={hidden ? 0.06 : weight} />;
      });
      break;
    }
    case "mlp":
      body = (
        <>
          <path d="M6 15 L16 7 L24 7 L34 15 L34 25 L24 33 L16 33 L6 25 Z" {...soft} />
          <path d="M6 20 H34" stroke={colour} strokeOpacity="0.4" />
          <circle cx="20" cy="20" r="2.4" fill={colour} />
        </>
      );
      break;
    case "moe":
      body = (
        <>
          <circle cx="7" cy="20" r="3" fill={colour} />
          {[0, 1, 2, 3].map((i) => (
            <g key={i}>
              <path d={`M10 20 L22 ${9 + i * 7.4}`} stroke={colour} strokeOpacity={i === 1 || i === 2 ? 0.9 : 0.2} strokeWidth="1" />
              <rect x="22" y={6 + i * 7.4} width="12" height="6" rx="1.5" fill={colour} fillOpacity={i === 1 || i === 2 ? 0.9 : 0.2} />
            </g>
          ))}
        </>
      );
      break;
    case "cls_token":
    case "take_first":
      body = [0, 1, 2, 3, 4].map((i) => <rect key={i} x={5 + i * 6.2} y="15" width="5" height="10" rx="1" fill={colour} fillOpacity={i === 0 ? 1 : 0.25} />);
      break;
    default:
      body = <rect x="9" y="9" width="22" height="22" rx="4" {...soft} />;
  }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="size-full text-ink" aria-hidden>
      {body}
    </svg>
  );
}
