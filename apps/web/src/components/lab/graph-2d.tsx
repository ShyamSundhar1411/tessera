"use client";

import { Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow, useReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { ChevronDown, ChevronRight, TriangleExclamationPoint } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { formatCount, type ChangeKind } from "tessera-engine";
import { buildDisplay, CATEGORY_COLOURS, formatShape, type Display, type DisplayNode } from "@/lib/display";
import { addNode, connect, disconnect, removeNode } from "@/lib/edit";
import { useLab, type SlotIndex } from "@/lib/store";
import { cn } from "@/lib/utils";

const STATUS_RING: Record<ChangeKind, string> = {
  same: "",
  changed: "ring-2 ring-changed",
  added: "ring-2 ring-added",
  removed: "ring-2 ring-removed",
};

type Data = DisplayNode & { slot: SlotIndex; shapeText: string; editing: boolean } & Record<string, unknown>;

function LayerNode({ data, selected }: NodeProps<Node<Data>>) {
  const colour = CATEGORY_COLOURS[data.category];
  return (
    <div
      className={cn(
        "relative flex h-[58px] w-[210px] items-stretch overflow-hidden rounded-lg border border-line bg-surface text-left shadow-sm",
        data.status && STATUS_RING[data.status],
        selected && "border-accent",
        data.error && "border-danger",
      )}
    >
      <Handle type="target" position={Position.Top} className={cn("!h-2 !w-2 !border-0 !bg-muted", !data.editing && "!opacity-0")} />
      <span className="w-1.5 flex-none" style={{ background: colour }} />
      <div className="min-w-0 flex-1 px-2.5 py-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-[12.5px] font-medium">{data.label}</span>
          {data.error && <TriangleExclamationPoint className="size-3.5 flex-none text-danger" />}
        </div>
        <div className="mt-0.5 flex items-center justify-between gap-2 font-mono text-[10.5px] text-muted">
          <span className="truncate">{data.op !== data.label ? data.op : ""} {data.shapeText && `→ ${data.shapeText}`}</span>
          {data.params > 0 && <span className="flex-none">{formatCount(data.params)}</span>}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className={cn("!h-2 !w-2 !border-0 !bg-muted", !data.editing && "!opacity-0")} />
    </div>
  );
}

function BlockNode({ data, selected }: NodeProps<Node<Data>>) {
  const toggle = useLab((state) => state.toggleExpanded);
  return (
    <div
      className={cn(
        "relative flex h-[66px] w-[230px] items-center gap-2 rounded-xl border border-line bg-surface-2 px-3 text-left",
        data.status && STATUS_RING[data.status],
        selected && "border-accent",
      )}
    >
      <Handle type="target" position={Position.Top} className="!h-2 !w-2 !border-0 !bg-muted !opacity-0" />
      <button type="button" onClick={() => toggle(data.slot, data.id)} className="grid size-7 flex-none place-items-center rounded-md border border-line bg-surface text-muted hover:text-ink" aria-label="Open block">
        <ChevronRight className="size-4" />
      </button>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-[13px] font-medium">{data.label}</span>
          {data.repeat > 1 && <span className="flex-none rounded bg-accent/15 px-1.5 font-mono text-[11px] text-accent">×{data.repeat}</span>}
        </div>
        <div className="mt-0.5 flex justify-between gap-2 font-mono text-[10.5px] text-muted">
          <span className="truncate">
            {data.size} layers {data.shapeText && `→ ${data.shapeText}`}
          </span>
          <span className="flex-none">{formatCount(data.params)}</span>
        </div>
      </div>
      {data.error && <TriangleExclamationPoint className="size-3.5 flex-none text-danger" />}
      <Handle type="source" position={Position.Bottom} className="!h-2 !w-2 !border-0 !bg-muted !opacity-0" />
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

/** Fits the view once a new layout arrives. */
function FitOnChange({ token }: { token: string }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const frame = requestAnimationFrame(() => fitView({ padding: 0.12, maxZoom: 1.1, duration: 250 }));
    return () => cancelAnimationFrame(frame);
  }, [token, fitView]);
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

  const nodes: Node<Data>[] = useMemo(
    () =>
      (display?.nodes ?? []).map((item) => ({
        id: item.id,
        type: item.open ? "container" : item.kind === "block" ? "block" : "layer",
        position: { x: item.x, y: item.y },
        ...(item.parent ? { parentId: item.parent, extent: "parent" as const } : {}),
        ...(item.open ? { style: { width: item.width, height: item.height }, selectable: false, zIndex: -1 } : {}),
        selected: selection?.slot === slot && selection.id === item.id,
        draggable: !item.open,
        connectable: editing && item.kind === "node",
        data: { ...item, slot, editing, shapeText: data ? formatShape(item.shape, data.model) : "" },
      })),
    [display, selection, slot, editing, data],
  );

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
        edges={edges}
        nodeTypes={nodeTypes}
        minZoom={0.05}
        maxZoom={2}
        nodesConnectable={editing}
        deleteKeyCode={editing ? ["Backspace", "Delete"] : null}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => select({ slot, id: node.id })}
        onNodeDoubleClick={(_, node) => {
          if (node.type === "block") toggle(slot, node.id);
        }}
        onPaneClick={() => select(null)}
        onConnect={(connection) => {
          if (connection.source && connection.target && realIds.has(connection.source) && realIds.has(connection.target)) edit(slot, (model) => connect(model, connection.source, connection.target));
        }}
        isValidConnection={(connection) => realIds.has(connection.source) && realIds.has(connection.target)}
        onNodesDelete={(deleted) => edit(slot, (model) => deleted.filter((node) => realIds.has(node.id)).reduce((m, node) => removeNode(m, node.id), model))}
        onEdgesDelete={(deleted) => edit(slot, (model) => deleted.reduce((m, edge) => disconnect(m, edge.source, edge.target), model))}
      >
        <FitOnChange token={`${data.model.name}|${data.expanded.join(",")}|${display ? "1" : "0"}`} />
        <Background gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap pannable zoomable position="bottom-right" nodeColor={(node) => CATEGORY_COLOURS[(node.data as unknown as Data).category] ?? "#888"} maskColor="color-mix(in srgb, var(--bg) 70%, transparent)" />
      </ReactFlow>
    </div>
  );
}
