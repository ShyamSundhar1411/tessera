import { ArrowRight, Box, Code, FlaskConical, GitCompareArrows, Network, Pencil } from "lucide-react";
import Link from "next/link";
import { analyze, CATALOG, formatCount } from "tessera-engine";
import { Mark } from "@/components/brand/mark";
import HeroScene from "@/components/landing/hero-scene";
import { SITE_URL } from "@/lib/site";

// The landing page: what Tessera does, shown with real numbers from the
// engine, so every figure on this page is computed, not written by hand.

const FEATURES = [
  {
    icon: Network,
    title: "Load any architecture",
    body: "Classic CNNs, small MLPs, an ONNX file, or any open language model's config.json from Hugging Face: Llama, Mistral, Qwen, Gemma, DeepSeek, Phi.",
  },
  {
    icon: Box,
    title: "See it in 2D and 3D",
    body: "A clean graph with blocks you can open, and a 3D view of the data itself: feature maps, kernel cones, neurons, token vectors and attention.",
  },
  {
    icon: GitCompareArrows,
    title: "Compare two side by side",
    body: "Parameters, compute, memory and KV cache, the settings that differ, and every changed block marked on both drawings.",
  },
  {
    icon: Pencil,
    title: "Rewire it",
    body: "Drag in layers, connect them, add a skip, change how often a block repeats. Shapes are checked as you go, and the text version updates with you.",
  },
  {
    icon: FlaskConical,
    title: "Train both and compare",
    body: "Small experiments in your browser, over several seeds, with accuracy, error rate, per-class F1, confusion matrices and whether the gap beats the noise.",
  },
  {
    icon: Code,
    title: "Export the code",
    body: "PyTorch modules built from your connections, and Colab notebooks that train the original and your version on a free GPU.",
  },
];

const PAIRS: [string, string, string][] = [
  ["mistral-7b", "mixtral-8x7b", "Dense layers to 8 experts"],
  ["llama-3.1-8b", "deepseek-v3", "Grouped-query to latent attention"],
  ["cifar-cnn", "cifar-cnn-skip", "One skip connection"],
];

export default function Home() {
  const pairs = PAIRS.map(([a, b, label]) => {
    const ea = CATALOG.find((entry) => entry.id === a)!;
    const eb = CATALOG.find((entry) => entry.id === b)!;
    const ma = analyze(ea.build());
    const mb = analyze(eb.build());
    return { label, a: ea.name, b: eb.name, pa: ma.params, pb: mb.params, activeB: mb.activeParams, href: `/lab/?a=${a}&b=${b}` };
  });
  const structured = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        "@id": `${SITE_URL}/#app`,
        name: "Tessera",
        url: `${SITE_URL}/lab/`,
        description:
          "A neural network architecture lab: load, compare, rewire, train and export architectures from small CNNs to large language models, in the browser.",
        applicationCategory: "DeveloperApplication",
        operatingSystem: "Any modern web browser",
        isAccessibleForFree: true,
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        author: { "@id": "https://shyamsundhar.com/#about" },
        creator: { "@id": "https://shyamsundhar.com/#about" },
      },
      { "@type": "Person", "@id": "https://shyamsundhar.com/#about", name: "Shyam Sundhar", url: "https://shyamsundhar.com" },
    ],
  };

  return (
    <div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structured).replace(/</g, "\\u003c") }} />
      <header className="mx-auto flex h-16 max-w-6xl items-center gap-2 px-6">
        <Mark className="size-7" />
        <span className="text-[16px] font-semibold tracking-tight">Tessera</span>
        <Link href="/lab/" className="ml-auto rounded-lg bg-ink px-3 py-1.5 text-[13.5px] font-medium text-bg hover:opacity-85">
          Open the lab
        </Link>
      </header>

      <section className="mx-auto grid max-w-6xl items-center gap-10 px-6 pt-14 pb-20 sm:pt-20 lg:grid-cols-[minmax(0,1fr)_minmax(0,32rem)]">
        <div>
          <p className="text-[13px] font-medium text-accent">Neural network architectures, hands on</p>
          <h1 className="mt-3 max-w-3xl text-4xl sm:text-[3.4rem]">
            Rewire a network. <span className="text-brand">See what changes.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-muted">
            Many breakthroughs came from changing how layers connect: a shortcut here, shared attention heads there, experts instead of one feed-forward.
            Tessera lets you load a real architecture, from a small CNN to an LLM, change its connections and find out what happens.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/lab/" className="inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-[15px] font-medium text-bg hover:opacity-85">
              Open the lab
              <ArrowRight className="size-4" />
            </Link>
            <a href="#how" className="rounded-xl border border-line px-4 py-2.5 text-[15px] font-medium hover:border-muted">
              What it does
            </a>
          </div>
          <p className="mt-4 text-[13px] text-muted">Free, runs in your browser, no account. Nothing you load leaves your machine.</p>
        </div>
        <HeroScene />
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-20">
        <h2 className="text-2xl sm:text-3xl">What changed between them</h2>
        <p className="mt-2 max-w-2xl text-muted">
          Three comparisons you can open in the lab. Every number is computed from the architecture, and the catalog matches each model&apos;s published
          parameter count.
        </p>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {pairs.map((pair) => (
            <Link key={pair.label} href={pair.href} className="group rounded-2xl border border-line bg-surface p-5 transition hover:border-muted">
              <p className="text-[12.5px] text-accent">{pair.label}</p>
              <p className="mt-2 text-[15px] font-medium">
                {pair.a} <span className="text-muted">→</span> {pair.b}
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-y-1 font-mono text-[12.5px]">
                <dt className="text-muted">Before</dt>
                <dd className="text-right">{formatCount(pair.pa)}</dd>
                <dt className="text-muted">After</dt>
                <dd className="text-right">{formatCount(pair.pb)}</dd>
                {pair.activeB !== pair.pb && (
                  <>
                    <dt className="text-muted">Active per token</dt>
                    <dd className="text-right">{formatCount(pair.activeB)}</dd>
                  </>
                )}
              </dl>
            </Link>
          ))}
        </div>
      </section>

      <section id="how" className="mx-auto max-w-6xl scroll-mt-16 px-6 pb-24">
        <h2 className="text-2xl sm:text-3xl">From config file to experiment</h2>
        <div className="mt-8 grid gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <div key={title}>
              <span className="grid size-9 place-items-center rounded-xl bg-surface-2 text-accent">
                <Icon className="size-4.5" />
              </span>
              <h3 className="mt-3 text-[16px]">{title}</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-muted">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="rounded-2xl border border-line bg-surface px-6 py-12 text-center">
          <h2 className="mx-auto max-w-2xl text-2xl sm:text-3xl">Start from a model you know, then change one thing.</h2>
          <p className="mt-3 text-muted">Add a skip connection, swap the attention, add experts. Then train both and see if it mattered.</p>
          <Link href="/lab/" className="mt-6 inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-[15px] font-medium text-bg hover:opacity-85">
            Open the lab
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-6 py-6 text-[13px] text-muted">
          <Mark className="size-5" />
          <span>Tessera</span>
          <span className="ml-auto">
            Designed and built by{" "}
            <a href="https://shyamsundhar.com" rel="author" className="text-ink hover:underline">
              Shyam Sundhar
            </a>
          </span>
        </div>
      </footer>
    </div>
  );
}
