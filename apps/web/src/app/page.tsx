import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { Mark } from "@/components/brand/mark";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-5xl flex-col px-6">
      <header className="flex h-16 items-center gap-2">
        <Mark className="size-7" />
        <span className="text-[16px] font-semibold tracking-tight">Tessera</span>
      </header>
      <section className="flex flex-1 flex-col justify-center pb-24">
        <p className="text-[13px] font-medium text-accent">Neural network architectures, hands on</p>
        <h1 className="mt-3 max-w-3xl text-4xl sm:text-6xl">
          Rewire a network. <span className="text-brand">See what changes.</span>
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
          Load anything from a small CNN to an LLM&apos;s config.json, see it in 2D and 3D, compare two side by side, change its connections, train both and export the PyTorch code.
        </p>
        <div className="mt-8">
          <Link href="/lab/" className="inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-[15px] font-medium text-bg hover:opacity-85">
            Open the lab
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </section>
    </main>
  );
}
