"use client";

import { Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow, useNodesState, useReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { ChevronDown, ChevronRight, TriangleExclamationPoint } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { formatCount, OPS, type ChangeKind, type Model } from "tessera-engine";
import { buildDisplay, CATEGORY_COLOURS, formatShape, type Display, type DisplayNode } from "@/lib/display";
import { addNode, connect, connectionEnd, disconnect, removeNode } from "@/lib/edit";
import { useLab, type SlotIndex } from "@/lib/store";
import { cn } from "@/lib/utils";
import { OpGlyph } from "./op-glyph";

const STATUS_RING: Record<ChangeKind, string> = {
  same: "",
  changed: "ring-2 ring-changed",
  added: "ring-2 ring-added",
  removed: "ring-2 ring-removed",
};

type Data = DisplayNode & { slot: SlotIndex; shapeText: string; editing: boolean } & Record<string, unknown>;

const HANDLE = "!h-3 !w-3 !border-2 !border-surface !bg-accent opacity-0 transition-opacity group-hover:opacity-100";

/** A causal attention layer draws its masked triangle; others draw a full matrix. */
const isCausal = (model: Model | undefined, id: string) => {
  const node = model?.nodes.find((item) => item.id === id);
  return !node || node.args.causal === undefined ? model?.kind === "language" : node.args.causal === true || node.args.causal === "true";
};

function LayerNode({ data, selected }: NodeProps<Node<Data>>) {
  const colour = CATEGORY_COLOURS[data.category];
  const model = useLab((state) => state.slots[data.slot]?.model);
  return (
    <div
      className={cn(
        "group relative flex h-[58px] w-[210px] items-center gap-2 rounded-xl border border-line bg-gradient-to-br from-surface to-surface-2 pr-2.5 pl-1.5 text-left shadow-sm transition-colors hover:border-muted",
        data.status && STATUS_RING[data.status],
        selected && "border-accent shadow-[0_0_0_3px_color-mix(in_oklab,var(--accent)_20%,transparent)]",
        data.error && "border-danger",
      )}
    >
      <Handle type="target" position={Position.Top} className={HANDLE} />
      <div className="size-[44px] flex-none rounded-lg p-0.5" style={{ background: `color-mix(in oklab, ${colour} 13%, transparent)` }}>
        <OpGlyph op={data.op} shape={data.shape} colour={colour} causal={isCausal(model, data.id)} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-[12.5px] font-medium">{data.label}</span>
          {data.error && <TriangleExclamationPoint className="size-3.5 flex-none text-danger" />}
        </div>
        <div className="mt-0.5 truncate font-mono text-[10.5px] text-muted">{data.shapeText && `→ ${data.shapeText}`}</div>
        <div className="flex items-center justify-between gap-2 font-mono text-[10px] text-muted">
          <span className="truncate" style={{ color: colour }}>{data.op !== data.label ? data.op : ""}</span>
          {data.params > 0 && <span className="flex-none">{formatCount(data.params)}</span>}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className={HANDLE} />
    </div>
  );
}

/** What a block is made of, most telling first: attention, then convolution, then the rest. */
const RANK = ["attention", "embedding", "conv", "mlp", "linear", "merge", "pool", "norm", "activation"];

function BlockNode({ data, selected }: NodeProps<Node<Data>>) {
  const toggle = useLab((state) => state.toggleExpanded);
  const model = useLab((state) => state.slots[data.slot]?.model);
  const members = useMemo(() => {
    if (!model) return [];
    const within = (groupId: string | undefined): boolean => {
      for (let g = groupId; g; g = model.groups.find((group) => group.id === g)?.parent) if (g === data.id) return true;
      return false;
    };
    return model.nodes.filter((node) => within(node.group));
  }, [model, data.id]);
  const lead = [...members].sort((a, b) => {
    const rank = (op: string) => {
      const index = RANK.indexOf(OPS[op]?.category ?? "");
      return index < 0 ? 99 : index;
    };
    return rank(a.op) - rank(b.op);
  })[0];
  const leadColour = lead ? CATEGORY_COLOURS[OPS[lead.op]?.category ?? "block"] : CATEGORY_COLOURS.block;
  const residual = members.some((node) => node.op === "add");
  return (
    <div className="group relative h-[66px] w-[230px]">
      {/* A repeated block shows its copies stacked behind it. */}
      {data.repeat > 1 && (
        <>
          <div className="absolute inset-0 translate-x-[8px] translate-y-[8px] rounded-xl border border-line bg-surface-2/50" />
          <div className="absolute inset-0 translate-x-[4px] translate-y-[4px] rounded-xl border border-line bg-surface-2/80" />
        </>
      )}
      <div
        className={cn(
          "relative flex h-full items-center gap-2 overflow-hidden rounded-xl border border-line bg-gradient-to-br from-surface-2 to-surface-3 pr-3 pl-1.5 text-left transition-colors hover:border-muted",
          data.status && STATUS_RING[data.status],
          selected && "border-accent",
        )}
      >
        <div className="relative size-[46px] flex-none rounded-lg p-0.5" style={{ background: `color-mix(in oklab, ${leadColour} 14%, transparent)` }}>
          <OpGlyph op={lead?.op} shape={data.shape} colour={leadColour} causal={lead ? isCausal(model, lead.id) : true} />
          {residual && (
            <svg viewBox="0 0 46 46" className="absolute inset-0" aria-hidden>
              <path d="M3 40 C -2 10, 48 10, 43 40" fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeDasharray="3 2" />
            </svg>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="truncate text-[13px] font-semibold">{data.label}</span>
            {data.repeat > 1 && <span className="flex-none rounded bg-accent/15 px-1.5 font-mono text-[11px] text-accent">×{data.repeat}</span>}
          </div>
          <div className="mt-0.5 flex justify-between gap-2 font-mono text-[10.5px] text-muted">
            <span className="truncate">{data.shapeText && `→ ${data.shapeText}`}</span>
            <span className="flex-none">{formatCount(data.params)}</span>
          </div>
          {/* Its layers in order, one segment each, in their colours. */}
          <div className="mt-1 flex h-1.5 gap-px overflow-hidden rounded-full" title={members.map((node) => node.label ?? node.op).join(" → ")}>
            {members.map((node) => (
              <span key={node.id} className="flex-1" style={{ background: CATEGORY_COLOURS[OPS[node.op]?.category ?? "block"] }} />
            ))}
          </div>
        </div>
        <button type="button" onClick={() => toggle(data.slot, data.id)} className="grid size-6 flex-none place-items-center rounded-md border border-line bg-surface text-muted hover:text-ink" aria-label="Open block">
          <ChevronRight className="size-3.5" />
        </button>
        {data.error && <TriangleExclamationPoint className="size-3.5 flex-none text-danger" />}
      </div>
      <Handle type="target" position={Position.Top} className={HANDLE} />
      <Handle type="source" position={Position.Bottom} className={HANDLE} />
    </div>
  );
}

function ContainerNode({ data }: NodeProps<Node<Data>>) {
  const toggle = useLab((state) => state.toggleExpanded);
  return (
    <div className={cn("h-full w-full rounded-2xl border border-dashed border-line bg-surface-2/40", data.status && STATUS_RING[data.status])}>
      <button type="button" onClick={() => toggle(data.slot, data.id)} className="flex w-full items-center gap-2 px-3 pt-2.5 text-left text-[12px] text-muted hover:text-ink">
        <ChevronDown className="size-3.5" />
        <span className="font-medium text-ink">{data.label}</span>
        {data.repeat > 1 && <span className="rounded bg-accent/15 px-1.5 font-mono text-[11px] text-accent">×{data.repeat}</span>}
        <span className="ml-auto font-mono text-[10.5px]">{formatCount(data.params)}</span>
      </button>
    </div>
  );
}

const nodeTypes = { layer: LayerNode, block: BlockNode, container: ContainerNode };

/** Fits the view when a new layout arrives, and again when the pane changes size (such as when B opens beside A). */
function FitOnChange({ token, container }: { token: string; container: React.RefObject<HTMLDivElement | null> }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const frame = requestAnimationFrame(() => fitView({ padding: 0.12, maxZoom: 1.1, duration: 250 }));
    return () => cancelAnimationFrame(frame);
  }, [token, fitView]);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    let last = element.clientWidth;
    const observer = new ResizeObserver(() => {
      if (Math.abs(element.clientWidth - last) < 40) return;
      last = element.clientWidth;
      fitView({ padding: 0.12, maxZoom: 1.1, duration: 200 });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [container, fitView]);
  return null;
}

export default function Graph2D({ slot }: { slot: SlotIndex }) {
  const data = useLab((state) => state.slots[slot]);
  const status = useLab((state) => state.diff?.status[slot]);
  const editing = useLab((state) => state.editing);
  const selection = useLab((state) => state.selection);
  const select = useLab((state) => state.select);
  const edit = useLab((state) => state.edit);
  const toggle = useLab((state) => state.toggleExpanded);
  const [display, setDisplay] = useState<Display | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const comparing = useLab((state) => !!state.slots[0] && !!state.slots[1]);

  useEffect(() => {
    if (!data) return;
    let live = true;
    void buildDisplay(data.model, data.analysis, data.expanded, status).then((result) => {
      if (live) setDisplay(result);
    });
    return () => {
      live = false;
    };
  }, [data, status]);

  const realIds = useMemo(() => new Set(data?.model.nodes.map((node) => node.id) ?? []), [data]);

  const laidOut: Node<Data>[] = useMemo(
    () =>
      (display?.nodes ?? []).map((item) => ({
        id: item.id,
        type: item.open ? "container" : item.kind === "block" ? "block" : "layer",
        position: { x: item.x, y: item.y },
        ...(item.parent ? { parentId: item.parent, extent: "parent" as const } : {}),
        ...(item.open ? { style: { width: item.width, height: item.height }, selectable: false, zIndex: -1 } : {}),
        draggable: !item.open,
        // Any layer or closed block can be wired by dragging from its bottom dot; Undo takes it back.
        connectable: !item.open,
        data: { ...item, slot, editing, shapeText: data ? formatShape(item.shape, data.model) : "" },
      })),
    [display, slot, editing, data],
  );

  // The layout places every box; dragging one moves it until the model or open blocks change.
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<Data>>([]);
  useEffect(() => setNodes(laidOut), [laidOut, setNodes]);
  // Selection made elsewhere (the inspector, the compare list) shows on the canvas without moving anything.
  useEffect(() => {
    setNodes((current) => current.map((node) => ({ ...node, selected: selection?.slot === slot && selection.id === node.id })));
  }, [selection, slot, setNodes]);

  const edges: Edge[] = useMemo(
    () =>
      (display?.edges ?? []).map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "default",
        animated: edge.skip,
        deletable: editing && realIds.has(edge.source) && realIds.has(edge.target),
        markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: edge.skip ? "var(--accent)" : "var(--muted)" },
        style: { stroke: edge.skip ? "var(--accent)" : "var(--muted)", strokeWidth: edge.skip ? 2 : 1.4 },
      })),
    [display, editing, realIds],
  );

  if (!data) return null;
  return (
    <div
      ref={container}
      className="h-full w-full"
      onDragOver={(event) => {
        if (editing && event.dataTransfer.types.includes("application/x-tessera-op")) event.preventDefault();
      }}
      onDrop={(event) => {
        const op = event.dataTransfer.getData("application/x-tessera-op");
        if (!op || !editing) return;
        event.preventDefault();
        // A dropped layer reads from the selected layer, so chains build quickly.
        const after = selection?.slot === slot && realIds.has(selection.id) ? selection.id : undefined;
        let created = "";
        edit(slot, (model) => {
          const result = addNode(model, op, after, after ? model.nodes.find((node) => node.id === after)?.group : undefined);
          created = result.id;
          return result.model;
        });
        if (created) select({ slot, id: created });
      }}
    >
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        // Trackpads: two fingers move around, pinch zooms. Mice: drag to move, Ctrl or ⌘ and scroll to zoom.
        panOnScroll
        zoomOnScroll={false}
        zoomOnPinch
        panOnDrag
        selectionOnDrag={false}
        edges={edges}
        nodeTypes={nodeTypes}
        minZoom={0.05}
        maxZoom={2}
        nodesConnectable
        connectionRadius={36}
        deleteKeyCode={editing ? ["Backspace", "Delete"] : null}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => select({ slot, id: node.id })}
        onNodeDoubleClick={(_, node) => {
          if (node.type === "block") toggle(slot, node.id);
        }}
        onPaneClick={() => select(null)}
        onConnect={(connection) => {
          edit(slot, (model) => {
            const source = connectionEnd(model, connection.source, "source");
            const target = connectionEnd(model, connection.target, "target");
            return source && target ? connect(model, source, target) : model;
          });
        }}
        isValidConnection={(connection) => connection.source !== connection.target}
        onNodesDelete={(deleted) => edit(slot, (model) => deleted.filter((node) => realIds.has(node.id)).reduce((m, node) => removeNode(m, node.id), model))}
        onEdgesDelete={(deleted) => edit(slot, (model) => deleted.reduce((m, edge) => disconnect(m, edge.source, edge.target), model))}
      >
        <FitOnChange container={container} token={`${data.model.name}|${data.expanded.join(",")}|${display ? "1" : "0"}`} />
        <Background gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} position="bottom-left" />
        {!comparing && <MiniMap className="max-md:!hidden" pannable zoomable position="bottom-right" nodeColor={(node) => CATEGORY_COLOURS[(node.data as unknown as Data).category] ?? "#888"} maskColor="color-mix(in srgb, var(--bg) 70%, transparent)" />}
      </ReactFlow>
    </div>
  );
}
