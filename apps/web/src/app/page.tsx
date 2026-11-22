import { ArrowRight, Code2, Dumbbell, GraduationCap, Microscope, Network, Play, Presentation, Route, Sigma, Wrench } from "lucide-react";
import Link from "next/link";
import { analyze, CATALOG, formatBytes, formatCount, OPS, PAPER_TIMELINE, type Model } from "tessera-engine";
import { Mark } from "@/components/brand/mark";
import HeroScene from "@/components/landing/hero-scene";
import Reveal from "@/components/landing/reveal";
import { OpGlyph } from "@/components/lab/op-glyph";
import Tex from "@/components/lab/tex";
import { CATEGORY_COLOURS } from "@/lib/display";
import { explain } from "@/lib/equations";
import { gradientPaths, receptiveField, scaling } from "@/lib/maths";
import { SITE_URL } from "@/lib/site";

// The landing page: what Tessera does, shown with real numbers and real
// equations from the engine, so every figure here is computed, not typed.

const build = (id: string) => CATALOG.find((entry) => entry.id === id)!.build();

/** One layer's equation as the trace shows it. */
function equationOf(model: Model, op: string) {
  const analysis = analyze(model);
  const node = model.nodes.find((item) => item.op === op)!;
  const shape = (id: string) => analysis.info.get(id)!.shape;
  return { model: model.name, node, ...explain(node, node.inputs.map(shape), shape(node.id)) };
}

const MODES = [
  { icon: Network, title: "Architecture", body: "Every layer drawn as what it does, in 2D, and the tensors themselves in 3D. Open blocks, drag a layer onto another to add a skip, edit the text form." },
  { icon: Route, title: "Trace", body: "Type a sentence or draw a digit and follow it through every layer: the equation with this model's sizes, a plain explanation, and the real numbers." },
  { icon: Sigma, title: "Maths", body: "From the wiring alone: how far gradients travel, how much of the image each output sees, where the compute goes, what longer context costs." },
  { icon: Dumbbell, title: "Train", body: "A small version of each model learns a task picked from its wiring, on your GPU, over several seeds, so a difference can be told from luck." },
  { icon: Play, title: "Run", body: "Real published weights where they exist: GPT-2 writes text token by token, ResNet names your photo. Or run the model you just trained." },
  { icon: Code2, title: "Export", body: "PyTorch modules with exactly the same parameter count, and a Colab notebook that trains your version against the original on a free GPU." },
];

const AUDIENCES = [
  { icon: Microscope, title: "Researchers", body: "Try a connection idea on a real architecture in minutes, check the shapes and the cost, train both, and take the PyTorch code with you." },
  { icon: GraduationCap, title: "Students", body: "See what attention, residuals and normalisation actually do to numbers, layer by layer, with the maths beside the values." },
  { icon: Presentation, title: "Teachers", body: "Show the Transformer from the 2017 paper running live, or open a comparison link that lands straight in the right view." },
  { icon: Wrench, title: "Engineers", body: "Compare memory, KV cache and compute of two model configs before you download a single weight." },
];

const FAQ = [
  { q: "Is it free?", a: "Yes. It runs entirely in your browser, with no account and no server doing the work." },
  { q: "Does anything I load leave my machine?", a: "No. Configs, ONNX files, photos and text stay on the page. The only downloads are public model weights from Hugging Face, when you ask for them." },
  { q: "Which models can I load?", a: "The catalog has language models (Llama, Mistral, Mixtral, Qwen, DeepSeek-V3, Gemma, Phi, GPT-2), the original Transformer, BERT-style encoders, vision models from AlexNet to ViT, and small classics. You can also paste any open model's config.json, or load an ONNX file." },
  { q: "Are the numbers right?", a: "Every catalog model matches its published parameter count, exported PyTorch builds with the same count, and GPT-2 and the ResNets produce the same outputs as PyTorch." },
  { q: "Can it train a large language model?", a: "It trains a small version with the same wiring (attention kind, head sharing, gating, experts) so you can compare designs in seconds. The Colab export scales the experiment up." },
];

export default function Home() {
  const resnet = build("resnet50");
  const resnetAnalysis = analyze(resnet);
  const paths = gradientPaths(resnet, resnetAnalysis);
  const field = receptiveField(resnet, resnetAnalysis)!;
  const llama = build("llama-3.1-8b");
  const llamaScale = scaling(llama, analyze(llama))!;
  const deepseek = build("deepseek-v3");
  const deepseekScale = scaling(deepseek, analyze(deepseek))!;
  const equations = [equationOf(build("transformer-base"), "attention"), equationOf(resnet, "conv2d"), equationOf(build("gpt2"), "layernorm")];
  const strip = ["transformer-base", "gpt2", "llama-3.1-8b", "deepseek-v3", "mixtral-8x7b", "qwen3-30b-a3b", "gemma-2-9b", "bert-base", "vit-b16", "resnet50", "alexnet", "unet", "densenet121", "mobilenet-v2", "lenet5"].map((id) => {
    const entry = CATALOG.find((item) => item.id === id)!;
    return { id, name: entry.name, params: analyze(entry.build()).params };
  });
  const pairs = (
    [
      ["mistral-7b", "mixtral-8x7b", "Dense layers to 8 experts"],
      ["llama-3.1-8b", "deepseek-v3", "Grouped-query to latent attention"],
      ["cifar-cnn", "cifar-cnn-skip", "One skip connection"],
    ] as const
  ).map(([a, b, label]) => {
    const ma = analyze(build(a));
    const mb = analyze(build(b));
    return { label, a: CATALOG.find((e) => e.id === a)!.name, b: CATALOG.find((e) => e.id === b)!.name, pa: ma.params, pb: mb.params, activeB: mb.activeParams, href: `/lab/?a=${a}&b=${b}` };
  });
  const papers = PAPER_TIMELINE.filter((entry) => ["lenet5", "alexnet", "resnet50", "unet", "transformer-base", "bert-base", "gpt2", "vit-b16", "mixtral-8x7b", "deepseek-v3"].includes(entry.id));

  const structured = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        "@id": `${SITE_URL}/#app`,
        name: "Tessera",
        url: `${SITE_URL}/lab/`,
        description: "A neural network architecture lab: load, trace, compare, rewire, train and export architectures from small CNNs to large language models, in the browser.",
        applicationCategory: "DeveloperApplication",
        operatingSystem: "Any modern web browser",
        isAccessibleForFree: true,
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        author: { "@id": "https://shyamsundhar.com/#about" },
        creator: { "@id": "https://shyamsundhar.com/#about" },
      },
      { "@type": "Person", "@id": "https://shyamsundhar.com/#about", name: "Shyam Sundhar", url: "https://shyamsundhar.com" },
      { "@type": "FAQPage", mainEntity: FAQ.map(({ q, a }) => ({ "@type": "Question", name: q, acceptedAnswer: { "@type": "Answer", text: a } })) },
    ],
  };

  const heading = (eyebrow: string, title: string, sub?: string) => (
    <Reveal>
      <p className="text-[13px] font-medium text-accent">{eyebrow}</p>
      <h2 className="mt-1.5 max-w-3xl text-2xl text-balance sm:text-3xl">{title}</h2>
      {sub && <p className="mt-2 max-w-2xl text-muted">{sub}</p>}
    </Reveal>
  );

  return (
    <div className="overflow-x-clip">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structured).replace(/</g, "\\u003c") }} />
      <header className="mx-auto flex h-16 max-w-6xl items-center gap-2 px-6">
        <Mark className="size-7" />
        <span className="text-[16px] font-semibold tracking-tight">Tessera</span>
        <nav className="ml-6 hidden gap-5 text-[13.5px] text-muted md:flex">
          <a href="#trace" className="hover:text-ink">Trace</a>
          <a href="#papers" className="hover:text-ink">Papers</a>
          <a href="#maths" className="hover:text-ink">Maths</a>
          <a href="#faq" className="hover:text-ink">FAQ</a>
        </nav>
        <Link href="/lab/" className="ml-auto rounded-lg bg-ink px-3 py-1.5 text-[13.5px] font-medium text-bg hover:opacity-85">
          Open the lab
        </Link>
      </header>

      {/* Hero */}
      <section className="mx-auto grid max-w-6xl items-center gap-10 px-6 pt-12 pb-14 sm:pt-20 lg:grid-cols-[minmax(0,1fr)_minmax(0,32rem)]">
        <div>
          <p className="rise text-[13px] font-medium text-accent">A lab for neural network architectures</p>
          <h1 className="rise mt-3 max-w-3xl text-4xl text-balance sm:text-[3.6rem]" style={{ animationDelay: "80ms" }}>
            See inside any neural network. <span className="text-brand">Then rewire it.</span>
          </h1>
          <p className="rise mt-6 max-w-2xl text-lg leading-relaxed text-muted" style={{ animationDelay: "160ms" }}>
            Load a real architecture, from LeNet to DeepSeek-V3. Follow your own input through every layer with the maths written out. Drag one layer onto another to add a skip, then train both versions and see if it mattered.
          </p>
          <div className="rise mt-8 flex flex-wrap gap-3" style={{ animationDelay: "240ms" }}>
            <Link href="/lab/" className="inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-[15px] font-medium text-bg hover:opacity-85">
              Open the lab
              <ArrowRight className="size-4" />
            </Link>
            <Link href="/lab/?a=transformer-base&mode=trace" className="rounded-xl border border-line px-4 py-2.5 text-[15px] font-medium hover:border-muted">
              Trace the 2017 Transformer
            </Link>
          </div>
          <p className="rise mt-4 text-[13px] text-muted" style={{ animationDelay: "300ms" }}>
            Free, in your browser, no account. Nothing you load leaves your machine.
          </p>
        </div>
        <div className="rise" style={{ animationDelay: "200ms" }}>
          <HeroScene />
        </div>
      </section>

      {/* The catalog, looping */}
      <section className="marquee-wrap border-y border-line bg-surface py-4" aria-label="Architectures in the catalog">
        <div className="marquee gap-3">
          {[...strip, ...strip].map((item, i) => (
            <Link key={i} href={`/lab/?a=${item.id}`} className="flex flex-none items-baseline gap-2 rounded-full border border-line px-4 py-1.5 text-[13.5px] hover:border-muted" tabIndex={i < strip.length ? 0 : -1} aria-hidden={i >= strip.length}>
              {item.name}
              <span className="font-mono text-[11.5px] text-muted">{formatCount(item.params)}</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Trace: real equations */}
      <section id="trace" className="mx-auto mt-24 max-w-6xl scroll-mt-20 px-6">
        {heading("Trace", "Every layer, written out as maths with the real sizes", "Not a generic diagram: the equation for this layer of this model, its numbers substituted, and a sentence on what it does. In the lab, the values your input produced sit right under it.")}
        <div className="mt-8 grid gap-4 lg:grid-cols-3">
          {equations.map((item, index) => {
            const colour = CATEGORY_COLOURS[OPS[item.node.op].category];
            return (
              <Reveal key={item.node.op} delay={index * 110} className="flex flex-col rounded-2xl border border-line bg-surface p-5">
                <div className="flex items-center gap-3">
                  <div className="size-11 flex-none rounded-lg p-0.5" style={{ background: `color-mix(in oklab, ${colour} 14%, transparent)` }}>
                    <OpGlyph op={item.node.op} shape={[]} colour={colour} causal={false} />
                  </div>
                  <div>
                    <p className="text-[14px] font-semibold">{item.node.label ?? item.node.op}</p>
                    <p className="text-[12px] text-muted">{item.model}</p>
                  </div>
                </div>
                <div className="mt-4 text-[14px]">
                  <Tex tex={item.tex} display />
                </div>
                {item.numbers && (
                  <div className="mt-2 rounded-lg bg-surface-2 px-2 text-[12.5px] text-muted">
                    <Tex tex={item.numbers} display />
                  </div>
                )}
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted">{item.words}</p>
              </Reveal>
            );
          })}
        </div>
      </section>

      {/* Real weights */}
      <section className="mx-auto mt-24 grid max-w-6xl items-center gap-10 px-6 lg:grid-cols-[1fr_1.1fr]">
        <Reveal>
          <p className="text-[13px] font-medium text-accent">Real weights, on your GPU</p>
          <h2 className="mt-1.5 text-2xl text-balance sm:text-3xl">The actual GPT-2 and ResNet, not a toy copy</h2>
          <p className="mt-3 text-muted">
            One click downloads OpenAI&apos;s GPT-2 or torchvision&apos;s ResNets from Hugging Face into the same graph you see on the canvas. GPT-2 writes text and shows the five tokens it weighed at every step. ResNet names your photo, and the trace shows what every one of its layers saw.
          </p>
        </Reveal>
        <Reveal delay={120} className="grid grid-cols-2 gap-3">
          {[
            ["124M", "GPT-2 parameters running in the browser"],
            ["0.01", "largest gap from PyTorch's next-token scores"],
            ["3 decimals", "ResNet logits match torchvision on a test photo"],
            ["2 s", "to start again later: the browser keeps the weights"],
          ].map(([value, label]) => (
            <div key={label} className="rounded-2xl border border-line bg-surface p-5">
              <p className="font-mono text-2xl font-semibold text-brand">{value}</p>
              <p className="mt-1 text-[13px] leading-snug text-muted">{label}</p>
            </div>
          ))}
        </Reveal>
      </section>

      {/* Papers */}
      <section id="papers" className="mx-auto mt-24 max-w-6xl scroll-mt-20 px-6">
        {heading("Papers", "The ideas that changed how layers connect", `${PAPER_TIMELINE.length} landmark papers from 1998 to today, each one a model you can open, trace and take apart.`)}
        <ol className="scroll-quiet mt-8 flex snap-x gap-4 overflow-x-auto pb-4">
          {papers.map((entry, index) => (
            <Reveal as="li" key={entry.id} delay={(index % 4) * 80} className="w-72 flex-none snap-start">
              <Link href={`/lab/?a=${entry.id}`} className="flex h-full flex-col rounded-2xl border border-line bg-surface p-5 transition hover:border-accent">
                <span className="font-mono text-[13px] font-semibold text-accent">{entry.paper!.year}</span>
                <span className="mt-1 text-[15px] font-semibold">{entry.name}</span>
                <span className="mt-1 text-[12px] text-muted">{entry.paper!.title}</span>
                <span className="mt-3 flex-1 text-[13px] leading-relaxed">{entry.paper!.idea}</span>
                <span className="mt-4 inline-flex items-center gap-1 text-[12.5px] text-accent">
                  Open it <ArrowRight className="size-3.5" />
                </span>
              </Link>
            </Reveal>
          ))}
        </ol>
      </section>

      {/* Maths */}
      <section id="maths" className="mx-auto mt-24 max-w-6xl scroll-mt-20 px-6">
        {heading("Maths", "Facts you get from the wiring, before any training", "Worked out from the architecture alone and shown for any model you load. A few of them:")}
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <Reveal className="rounded-2xl border border-line bg-surface p-5">
            <p className="text-[13px] text-muted">ResNet-50: layers a gradient passes through</p>
            <p className="mt-2 font-mono text-3xl font-semibold">
              {paths.longest} <span className="text-muted">or</span> <span className="text-brand">{paths.shortest}</span>
            </p>
            <p className="mt-2 text-[13px] leading-relaxed text-muted">{paths.skips} skip connections give the error a route through only {paths.shortest} layers. That is why 50 layers train at all.</p>
          </Reveal>
          <Reveal delay={110} className="rounded-2xl border border-line bg-surface p-5">
            <p className="text-[13px] text-muted">ResNet-50: what one final output sees</p>
            <p className="mt-2 font-mono text-3xl font-semibold">
              {field[field.length - 1].field}×{field[field.length - 1].field}
              <span className="text-[15px] text-muted"> px</span>
            </p>
            <p className="mt-2 text-[13px] leading-relaxed text-muted">Its receptive field is wider than the 224 px image itself: every value at the end can draw on the whole picture.</p>
          </Reveal>
          <Reveal delay={220} className="rounded-2xl border border-line bg-surface p-5">
            <p className="text-[13px] text-muted">Key/value cache for a 128K-token context</p>
            <p className="mt-2 font-mono text-2xl font-semibold whitespace-nowrap sm:text-[1.7rem]">
              {formatBytes(llamaScale.kvBytes * 131072)} <span className="text-[15px] text-muted">vs</span> <span className="text-brand">{formatBytes(deepseekScale.kvBytes * 131072)}</span>
            </p>
            <p className="mt-2 text-[13px] leading-relaxed text-muted">Llama 3.1 8B against DeepSeek-V3, which is 80 times bigger. Latent attention caches a small summary instead of every head&apos;s keys and values.</p>
          </Reveal>
        </div>
      </section>

      {/* Comparisons */}
      <section className="mx-auto mt-24 max-w-6xl px-6">
        {heading("Compare", "One change, side by side", "Open a comparison and every mode shows both: the drawings with changed blocks marked, the trace, the maths and the training curves.")}
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {pairs.map((pair, index) => (
            <Reveal key={pair.label} delay={index * 100}>
              <Link href={pair.href} className="block h-full rounded-2xl border border-line bg-surface p-5 transition hover:border-muted">
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
            </Reveal>
          ))}
        </div>
      </section>

      {/* Modes */}
      <section className="mx-auto mt-24 max-w-6xl px-6">
        {heading("The lab", "Six ways to look at one architecture")}
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MODES.map(({ icon: Icon, title, body }, index) => (
            <Reveal key={title} delay={(index % 3) * 90} className="rounded-2xl border border-line bg-surface p-5">
              <span className="grid size-10 place-items-center rounded-xl bg-accent/15 text-accent">
                <Icon className="size-5" />
              </span>
              <h3 className="mt-3 text-[16px]">{title}</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-muted">{body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Who */}
      <section className="mx-auto mt-24 max-w-6xl px-6">
        {heading("Who it is for", "For anyone who wants to know what a network really does")}
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {AUDIENCES.map(({ icon: Icon, title, body }, index) => (
            <Reveal key={title} delay={index * 80} className="rounded-2xl border border-line p-5">
              <Icon className="size-5 text-accent" />
              <h3 className="mt-3 text-[15px]">{title}</h3>
              <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">{body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="mx-auto mt-24 max-w-3xl scroll-mt-20 px-6">
        {heading("FAQ", "Questions people ask")}
        <div className="mt-6 divide-y divide-line rounded-2xl border border-line bg-surface">
          {FAQ.map(({ q, a }) => (
            <details key={q} className="group px-5 py-4">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-[15px] font-medium">
                {q}
                <span className="text-muted transition group-open:rotate-45">+</span>
              </summary>
              <p className="mt-2 text-[14px] leading-relaxed text-muted">{a}</p>
            </details>
          ))}
        </div>
      </section>

      {/* Closing */}
      <section className="mx-auto mt-24 max-w-6xl px-6 pb-24">
        <Reveal className="relative overflow-hidden rounded-3xl border border-line bg-surface px-6 py-14 text-center">
          <div className="pointer-events-none absolute inset-0 opacity-40" style={{ background: "radial-gradient(60% 80% at 50% 0%, color-mix(in oklab, var(--accent) 30%, transparent), transparent)" }} />
          <h2 className="relative mx-auto max-w-2xl text-2xl text-balance sm:text-4xl">Start from a model you know. Change one connection.</h2>
          <p className="relative mt-3 text-muted">Then trace it, train it against the original, and take the code.</p>
          <div className="relative mt-7 flex flex-wrap justify-center gap-3">
            <Link href="/lab/" className="inline-flex items-center gap-2 rounded-xl bg-ink px-5 py-3 text-[15px] font-medium text-bg hover:opacity-85">
              Open the lab
              <ArrowRight className="size-4" />
            </Link>
            <Link href="/lab/?a=gpt2&mode=run" className="rounded-xl border border-line px-5 py-3 text-[15px] font-medium hover:border-muted">
              Run the real GPT-2
            </Link>
          </div>
        </Reveal>
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
