"use client";

import { ChevronDown, ChevronRight, Trash } from "lucide-react";
import { formatCount, OPS, type ArgValue } from "tessera-engine";
import { formatShape } from "@/lib/display";
import { removeNode, setRepeat, updateNode } from "@/lib/edit";
import { useLab } from "@/lib/store";

const field = "w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-[12px] outline-none focus:border-accent";

/** Settings each operation offers in the inspector, beyond whatever the node already has. */
const ARG_HINTS: Record<string, string[]> = {
  conv2d: ["out", "k", "s", "p", "groups", "bias"],
  linear: ["out", "bias"],
  maxpool: ["k", "s", "p"],
  avgpool: ["k", "s", "p"],
  dropout: ["rate"],
  concat: ["axis"],
  embedding: ["vocab", "dim"],
  posembed: ["max"],
  attention: ["heads", "kv", "head_dim", "window", "qkv_bias", "o_bias", "qk_norm"],
  mla: ["heads", "q_rank", "kv_rank", "qk_nope_dim", "qk_rope_dim", "v_dim"],
  mlp: ["ff", "act", "bias"],
  moe: ["experts", "topk", "ff", "shared", "shared_ff"],
  lmhead: ["vocab", "tied"],
  layernorm: ["bias"],
};
const BOOLEAN_ARGS = new Set(["bias", "qkv_bias", "o_bias", "qk_norm", "tied", "shared_gate", "router_bias"]);

const parseValue = (raw: string): ArgValue | undefined => {
  if (raw.trim() === "") return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : raw.trim();
};

export default function Inspector() {
  const selection = useLab((state) => state.selection);
  const slots = useLab((state) => state.slots);
  const editing = useLab((state) => state.editing);
  const edit = useLab((state) => state.edit);
  const select = useLab((state) => state.select);
  const toggle = useLab((state) => state.toggleExpanded);

  const data = selection ? slots[selection.slot] : null;
  if (!selection || !data)
    return (
      <div className="p-4 text-[13px] leading-relaxed text-muted">
        <p className="font-medium text-ink">Nothing selected</p>
        <p className="mt-1">Click a layer or block to see its shapes, parameters and compute, and to change its settings.</p>
        {data === null && <p className="mt-3">Turn on <span className="text-ink">Edit</span> to drag in new layers and rewire connections.</p>}
      </div>
    );

  const { model, analysis } = data;
  const node = model.nodes.find((item) => item.id === selection.id);
  const group = node ? undefined : model.groups.find((item) => item.id === selection.id);

  if (group) {
    const members = model.nodes.filter((item) => {
      let id = item.group;
      while (id) {
        if (id === group.id) return true;
        id = model.groups.find((g) => g.id === id)?.parent;
      }
      return false;
    });
    const params = members.reduce((sum, item) => sum + (analysis.info.get(item.id)?.params ?? 0) * (analysis.info.get(item.id)?.times ?? 1), 0);
    const ops = new Map<string, number>();
    for (const item of members) ops.set(item.op, (ops.get(item.op) ?? 0) + 1);
    const open = data.expanded.includes(group.id);
    return (
      <div className="space-y-4 p-4 text-[13px]">
        <div>
          <p className="text-[12px] text-muted">Block · {group.kind}</p>
          <h3 className="mt-0.5 text-[16px]">{group.label}</h3>
        </div>
        <button type="button" onClick={() => toggle(selection.slot, group.id)} className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1 text-[12.5px] hover:border-muted">
          {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          {open ? "Close block" : "Open block"}
        </button>
        <dl className="grid grid-cols-2 gap-y-1.5 text-[12.5px]">
          <dt className="text-muted">Parameters</dt>
          <dd className="text-right font-mono">{formatCount(params)}</dd>
          <dt className="text-muted">Layers inside</dt>
          <dd className="text-right font-mono">{members.length}</dd>
        </dl>
        <label className="block">
          <span className="text-[12px] text-muted">Repeats</span>
          <input type="number" min={1} value={group.repeat ?? 1} onChange={(event) => edit(selection.slot, (m) => setRepeat(m, group.id, Number(event.target.value) || 1))} className={field} />
          <span className="mt-1 block text-[11.5px] text-muted">A repeated block must keep its input shape. Changing the count rescales parameters and compute.</span>
        </label>
        <div>
          <p className="mb-1 text-[12px] text-muted">Made of</p>
          <ul className="space-y-0.5 font-mono text-[12px]">
            {[...ops].map(([op, count]) => (
              <li key={op} className="flex justify-between">
                <span>{op}</span>
                <span className="text-muted">×{count}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  if (!node) return null;
  const info = analysis.info.get(node.id);
  const spec = OPS[node.op];
  const keys = [...new Set([...(ARG_HINTS[node.op] ?? []), ...Object.keys(node.args).filter((key) => key !== "shape" && key !== "rope")])];
  const inputs = node.inputs.map((id) => ({ id, shape: analysis.info.get(id)?.shape ?? [] }));
  const readers = model.nodes.filter((item) => item.inputs.includes(node.id));

  return (
    <div className="space-y-4 p-4 text-[13px]">
      <div>
        <p className="text-[12px] text-muted">
          {spec?.category ?? "layer"} · <span className="font-mono">{node.id}</span>
        </p>
        <input
          value={node.label ?? ""}
          placeholder={node.op}
          onChange={(event) => edit(selection.slot, (m) => updateNode(m, node.id, { label: event.target.value }))}
          className="mt-0.5 w-full bg-transparent text-[16px] font-medium outline-none placeholder:text-ink"
          aria-label="Layer name"
        />
        {spec && <p className="mt-1 text-[12.5px] leading-relaxed text-muted">{spec.summary}</p>}
      </div>

      {info?.error && <p className="rounded-md border border-danger/50 bg-danger/10 px-2.5 py-2 text-[12.5px] text-danger">{info.error}</p>}

      <dl className="grid grid-cols-2 gap-y-1.5 text-[12.5px]">
        <dt className="text-muted">Output</dt>
        <dd className="text-right font-mono">{info?.shape.length ? formatShape(info.shape, model) : "-"}</dd>
        <dt className="text-muted">Parameters</dt>
        <dd className="text-right font-mono">
          {formatCount(info?.params ?? 0)}
          {info && info.times > 1 ? ` ×${info.times}` : ""}
        </dd>
        {info && info.activeParams !== info.params && (
          <>
            <dt className="text-muted">Active per token</dt>
            <dd className="text-right font-mono">{formatCount(info.activeParams)}</dd>
          </>
        )}
        <dt className="text-muted">Compute</dt>
        <dd className="text-right font-mono">{formatCount(info?.flops ?? 0)}FLOPs</dd>
      </dl>

      {node.op === "input" ? (
        <label className="block">
          <span className="text-[12px] text-muted">Shape</span>
          <input value={String(node.args.shape)} onChange={(event) => edit(selection.slot, (m) => updateNode(m, node.id, { args: { shape: event.target.value } }))} className={field} />
        </label>
      ) : (
        <div className="space-y-2">
          <p className="text-[12px] text-muted">Settings</p>
          {keys.length === 0 && <p className="text-[12px] text-muted">None.</p>}
          {keys.map((key) =>
            BOOLEAN_ARGS.has(key) ? (
              <label key={key} className="flex items-center justify-between gap-3 text-[12.5px]">
                <span className="font-mono">{key}</span>
                <input
                  type="checkbox"
                  checked={node.args[key] === true || node.args[key] === "true" || (node.args[key] === undefined && key === "bias" && node.op !== "mlp")}
                  onChange={(event) => edit(selection.slot, (m) => updateNode(m, node.id, { args: { [key]: event.target.checked } }))}
                  className="size-4 accent-[var(--accent)]"
                />
              </label>
            ) : (
              <label key={key} className="grid grid-cols-[110px_1fr] items-center gap-2 text-[12.5px]">
                <span className="font-mono">{key}</span>
                <input
                  defaultValue={node.args[key] === undefined ? "" : String(node.args[key])}
                  key={`${node.id}-${key}-${String(node.args[key])}`}
                  placeholder={key === spec?.main ? "required" : "default"}
                  onBlur={(event) => edit(selection.slot, (m) => updateNode(m, node.id, { args: { [key]: parseValue(event.target.value) } }))}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") (event.target as HTMLInputElement).blur();
                  }}
                  className={field}
                />
              </label>
            ),
          )}
        </div>
      )}

      <div className="space-y-1 text-[12.5px]">
        <p className="text-[12px] text-muted">Reads from</p>
        {inputs.length === 0 && node.op !== "input" && <p className="text-danger">Nothing. Connect an input in Edit mode.</p>}
        {inputs.map((input) => (
          <button key={input.id} type="button" onClick={() => select({ slot: selection.slot, id: input.id })} className="flex w-full justify-between rounded px-1 py-0.5 text-left font-mono hover:bg-surface-2">
            <span>{input.id}</span>
            <span className="text-muted">{formatShape(input.shape, model)}</span>
          </button>
        ))}
        <p className="pt-1 text-[12px] text-muted">Feeds</p>
        {readers.length === 0 && <p className="text-muted">Nothing (an output).</p>}
        {readers.map((reader) => (
          <button key={reader.id} type="button" onClick={() => select({ slot: selection.slot, id: reader.id })} className="block w-full rounded px-1 py-0.5 text-left font-mono hover:bg-surface-2">
            {reader.id}
          </button>
        ))}
      </div>

      {editing && node.op !== "input" && (
        <button
          type="button"
          onClick={() => {
            edit(selection.slot, (m) => removeNode(m, node.id));
            select(null);
          }}
          className="flex items-center gap-1.5 rounded-md border border-danger/50 px-2.5 py-1 text-[12.5px] text-danger hover:bg-danger/10"
        >
          <Trash className="size-3.5" />
          Remove layer
        </button>
      )}
    </div>
  );
}
