"use client";

import "katex/dist/katex.min.css";
import katex from "katex";
import { useMemo } from "react";
import { cn } from "@/lib/utils";

/** TeX typeset with KaTeX; display mode centres it on its own line. */
export default function Tex({ tex, display, className }: { tex: string; display?: boolean; className?: string }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: display, strict: false }), [tex, display]);
  return <span className={cn(display && "scroll-quiet block overflow-x-auto py-1", className)} dangerouslySetInnerHTML={{ __html: html }} />;
}
