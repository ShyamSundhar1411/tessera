"use client";

import { Billboard, Line, OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { CatmullRomCurve3, Color, Vector3, type Mesh } from "three";
import { formatCount, type ChangeKind, type Shape } from "tessera-engine";
import { buildDisplay, CATEGORY_COLOURS, formatShape, type Display, type DisplayNode } from "@/lib/display";
import { useLab, type SlotIndex } from "@/lib/store";

// The architecture as a 3D object. Positions come from the same layout as
// the 2D view; each layer becomes a block whose size follows its tensor:
// images are wide and tall by their spatial size and deep by their channels,
// sequences are wide by their feature size. Repeated blocks show as a stack.

const SCALE = 0.014;
const STATUS_COLOUR: Record<ChangeKind, string | null> = { same: null, changed: "#f7b955", added: "#3ddc97", removed: "#ff5d6c" };

const log = (n: number) => Math.log2(Math.max(2, n));

/** World size [x, y, z] for a tensor shape. Flow runs along z. */
function sizeFor(shape: Shape, block: boolean): [number, number, number] {
  if (shape.length === 3) {
    const [c, h] = shape;
    const side = 0.35 + log(h) * 0.16;
    return [side, side, 0.12 + log(c) * 0.05];
  }
  if (shape.length === 2) return [0.4 + log(shape[1]) * 0.14, 0.55, block ? 0.5 : 0.28];
  if (shape.length === 1) return [0.3 + log(shape[0]) * 0.11, 0.22, 0.22];
  return [0.6, 0.3, 0.3];
}

interface Placed {
  item: DisplayNode;
  position: Vector3;
  size: [number, number, number];
  colour: string;
}

function Layer({ placed, selected, onSelect, shapeText }: { placed: Placed; selected: boolean; onSelect: () => void; shapeText: string }) {
  const { item, position, size, colour } = placed;
  const ring = item.status ? STATUS_COLOUR[item.status] : null;
  const plates = item.kind === "block" ? Math.min(item.repeat, 10) : 1;
  const plateGap = item.kind === "block" ? size[1] * 0.38 : 0;
  return (
    <group position={position}>
      {Array.from({ length: plates }, (_, i) => (
        <mesh
          key={i}
          position={[0, size[1] / 2 + i * plateGap, 0]}
          castShadow
          onClick={(event) => {
            event.stopPropagation();
            onSelect();
          }}
        >
          <boxGeometry args={size} />
          <meshStandardMaterial
            color={colour}
            roughness={0.42}
            metalness={0.08}
            emissive={selected ? new Color(colour) : ring ? new Color(ring) : new Color("#000000")}
            emissiveIntensity={selected ? 0.55 : ring ? 0.35 : 0}
            transparent={plates > 1}
            opacity={plates > 1 ? 0.9 - i * 0.04 : 1}
            wireframe={!!item.error}
          />
        </mesh>
      ))}
      <Billboard position={[0, size[1] + (plates - 1) * plateGap + 0.32, 0]}>
        <Text fontSize={0.17} color="#eef0f5" anchorX="center" anchorY="bottom" outlineWidth={0.012} outlineColor="#0b0d12">
          {item.label}
          {item.repeat > 1 ? `  ×${item.repeat}` : ""}
        </Text>
        <Text position={[0, -0.04, 0]} fontSize={0.11} color="#9aa2b4" anchorX="center" anchorY="top" outlineWidth={0.008} outlineColor="#0b0d12">
          {[shapeText, item.params ? formatCount(item.params) : ""].filter(Boolean).join("  ·  ")}
        </Text>
      </Billboard>
    </group>
  );
}

/** A pulse travelling along a connection, so the direction of data is visible. */
function Pulse({ curve, colour, offset }: { curve: CatmullRomCurve3; colour: string; offset: number }) {
  const mesh = useRef<Mesh>(null);
  useFrame(({ clock }) => {
    const t = (clock.getElapsedTime() * 0.35 + offset) % 1;
    mesh.current?.position.copy(curve.getPointAt(t));
  });
  return (
    <mesh ref={mesh}>
      <sphereGeometry args={[0.05, 12, 12]} />
      <meshBasicMaterial color={colour} toneMapped={false} />
    </mesh>
  );
}

function Scene({ display, slot, model }: { display: Display; slot: SlotIndex; model: NonNullable<ReturnType<typeof useLab.getState>["slots"][0]>["model"] }) {
  const selection = useLab((state) => state.selection);
  const select = useLab((state) => state.select);
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const { placed, curves, centre, span } = useMemo(() => {
    const boxes = display.nodes.filter((item) => !item.open);
    const cx = display.width / 2;
    const cy = display.height / 2;
    const placed = new Map<string, Placed>();
    for (const item of boxes) {
      placed.set(item.id, {
        item,
        position: new Vector3((item.ax + item.width / 2 - cx) * SCALE, 0, (item.ay + item.height / 2 - cy) * SCALE),
        size: sizeFor(item.shape, item.kind === "block"),
        colour: CATEGORY_COLOURS[item.category],
      });
    }
    const curves = display.edges.flatMap((edge) => {
      const a = placed.get(edge.source);
      const b = placed.get(edge.target);
      if (!a || !b) return [];
      const start = a.position.clone().add(new Vector3(0, a.size[1] / 2, 0));
      const end = b.position.clone().add(new Vector3(0, b.size[1] / 2, 0));
      // Skip connections arch over the layers they jump.
      const lift = edge.skip ? 0.9 + start.distanceTo(end) * 0.18 : 0.05;
      const mid = start.clone().lerp(end, 0.5).add(new Vector3(edge.skip ? 0.6 : 0, lift, 0));
      return [{ edge, curve: new CatmullRomCurve3([start, mid, end]) }];
    });
    return { placed: [...placed.values()], curves, centre: new Vector3(0, 0, 0), span: Math.max(display.width, display.height) * SCALE };
  }, [display]);

  const platforms = display.nodes.filter((item) => item.open);
  return (
    <>
      <color attach="background" args={["#0b0d12"]} />
      <fog attach="fog" args={["#0b0d12", span * 0.9, span * 2.6]} />
      <hemisphereLight args={["#cbd6ff", "#1a1d26", 0.65]} />
      <directionalLight position={[span * 0.4, span * 0.8, span * 0.5]} intensity={1.5} castShadow />
      <ambientLight intensity={0.25} />
      <gridHelper args={[span * 1.6, 40, "#262b37", "#181c25"]} position={[0, -0.01, 0]} />
      {platforms.map((item) => (
        <mesh key={item.id} position={[(item.ax + item.width / 2 - display.width / 2) * SCALE, 0.005, (item.ay + item.height / 2 - display.height / 2) * SCALE]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[item.width * SCALE, item.height * SCALE]} />
          <meshStandardMaterial color="#f6782a" transparent opacity={0.07} />
        </mesh>
      ))}
      {curves.map(({ edge, curve }) => (
        <Line key={edge.id} points={curve.getPoints(24)} color={edge.skip ? "#f0569f" : "#56607a"} lineWidth={edge.skip ? 2.2 : 1.2} />
      ))}
      {!reduced && curves.map(({ edge, curve }, i) => <Pulse key={`p-${edge.id}`} curve={curve} colour={edge.skip ? "#f0569f" : "#b8f3ff"} offset={(i * 0.137) % 1} />)}
      {placed.map((item) => (
        <Layer
          key={item.item.id}
          placed={item}
          shapeText={formatShape(item.item.shape, model)}
          selected={selection?.slot === slot && selection.id === item.item.id}
          onSelect={() => select({ slot, id: item.item.id })}
        />
      ))}
      <OrbitControls target={centre} makeDefault enableDamping maxPolarAngle={Math.PI / 2.05} />
    </>
  );
}

export default function Graph3D({ slot }: { slot: SlotIndex }) {
  const data = useLab((state) => state.slots[slot]);
  const status = useLab((state) => state.diff?.status[slot]);
  const select = useLab((state) => state.select);
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

  if (!data || !display) return <div className="grid h-full place-items-center text-[13px] text-muted">Laying out…</div>;
  const span = Math.max(display.width, display.height) * SCALE;
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [span * 0.55, span * 0.6, span * 0.75], fov: 40, near: 0.05, far: span * 10 + 50 }} onPointerMissed={() => select(null)}>
      <Scene display={display} slot={slot} model={data.model} />
    </Canvas>
  );
}
