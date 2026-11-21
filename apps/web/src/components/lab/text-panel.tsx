"use client";

import { useEffect, useRef, useState } from "react";
import { useLab } from "@/lib/store";

/** The active architecture as text. Typing updates the drawing once the text reads cleanly. */
export default function TextPanel() {
  const active = useLab((state) => state.active);
  const slot = useLab((state) => state.slots[state.active]);
  const setText = useLab((state) => state.setText);
  const [draft, setDraft] = useState(slot?.text ?? "");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const gutter = useRef<HTMLDivElement>(null);

  // Edits made on the canvas flow back into the text.
  useEffect(() => {
    if (document.activeElement !== area.current && slot) setDraft(slot.text);
  }, [slot]);

  if (!slot) return null;
  const errors = [...slot.parseErrors.map((error) => ({ line: error.line, message: error.message })), ...slot.analysis.errors.filter((error) => error.line).map((error) => ({ line: error.line!, message: error.message }))];
  const lines = draft.split("\n").length;
  const errorLines = new Set(errors.map((error) => error.line));

  return (
    <div className="flex h-full flex-col">
      <div className="relative flex min-h-0 flex-1 font-mono text-[12px] leading-[1.6]">
        <div ref={gutter} aria-hidden="true" className="w-10 flex-none overflow-hidden border-r border-line py-3 pr-2 text-right text-muted/60 select-none">
          {Array.from({ length: lines }, (_, i) => (
            <div key={i} className={errorLines.has(i + 1) ? "text-danger" : undefined}>
              {i + 1}
            </div>
          ))}
        </div>
        <textarea
          ref={area}
          value={draft}
          spellCheck={false}
          aria-label="Architecture as text"
          onScroll={(event) => {
            if (gutter.current) gutter.current.scrollTop = event.currentTarget.scrollTop;
          }}
          onChange={(event) => {
            const value = event.target.value;
            setDraft(value);
            if (timer.current) clearTimeout(timer.current);
            timer.current = setTimeout(() => setText(active, value), 350);
          }}
          className="scroll-quiet min-h-0 flex-1 resize-none bg-transparent px-3 py-3 whitespace-pre outline-none"
        />
      </div>
      <div className="max-h-36 flex-none overflow-y-auto border-t border-line px-3 py-2 text-[12px]">
        {errors.length === 0 ? (
          <p className="text-muted">Reads cleanly. Write a line like <code className="text-ink">x = conv 64 k3 -&gt; bn -&gt; relu</code>, join with <code className="text-ink">add(a, b)</code>.</p>
        ) : (
          <ul className="space-y-0.5">
            {errors.slice(0, 8).map((error, i) => (
              <li key={i} className="text-danger">
                <span className="font-mono">line {error.line}</span>: {error.message}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
