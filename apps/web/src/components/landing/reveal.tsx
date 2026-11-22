"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface RevealProps {
  children: React.ReactNode;
  className?: string;
  /** Milliseconds to wait after coming into view, for staggering siblings. */
  delay?: number;
  as?: "div" | "section" | "li";
  id?: string;
}

/** Rises and fades in the first time it scrolls into view. */
export default function Reveal({ children, className, delay = 0, as: Tag = "div", id }: RevealProps) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        element.dataset.shown = "";
        observer.disconnect();
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <Tag ref={ref as never} id={id} data-reveal="" className={cn(className)} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </Tag>
  );
}
