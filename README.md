# Tessera

Rewire a neural network and see what changes. Load an architecture, from a small CNN to a large language model, see it in 2D and 3D, compare two side by side, change its connections, train both and export the PyTorch code. Everything runs in the browser.

## What it does

- **Load** from the catalog (Llama 3.1, Mistral, Mixtral, Qwen 2.5 and 3, DeepSeek-V3, Gemma 2 and 3, Phi-3, SmolLM2, GPT-2, VGG16, ResNet-18 and 50, MobileNetV2, EfficientNet-B0 and small models), any open model's `config.json` from Hugging Face (fetched or pasted), an `.onnx` file, or a blank start.
- **See** it as a 2D graph (blocks open and close, skip connections highlighted) or in 3D, where each stage is the tensor a layer hands on: feature maps with kernel cones, neuron columns, token vectors with attention arcs and experts.
- **Compare** A and B: parameters, active parameters, compute, 16-bit weight memory, KV cache per token, the settings that differ, and changed blocks marked on both drawings.
- **Rewire**: drag layers in, connect and disconnect, change settings and block repeats, undo. The architecture's text form stays in step and can be edited directly.
- **Train** both in the browser (TensorFlow.js) on MNIST, Fashion-MNIST or CIFAR-10 over several seeds: accuracy, error rate, top-5, per-class precision, recall and F1, confusion matrices, gradient size per layer, and whether the gap is bigger than seed-to-seed noise.
- **Run** pretrained ImageNet classifiers on your own photo (ONNX Runtime Web).
- **Export** PyTorch, the text format, JSON, or a Colab notebook that trains A and B over several seeds on a free GPU. Language models are shrunk to about 30M parameters with the same wiring and trained on TinyStories.

## Correctness

Every catalog model matches its published parameter count (image models to the parameter, language models within half a percent), and every exported PyTorch module builds with exactly the engine's count. Shrunk versions of each language model family run a forward pass in PyTorch.

## Layout

- `packages/engine`: the graph format, operations with shape, parameter and compute rules, analysis, the catalog, Hugging Face and ONNX import, the text format, diff, shrink, PyTorch and notebook export. `bun test` runs its tests.
- `packages/lab`: browser training (TensorFlow.js) and pretrained inference (ONNX Runtime Web).
- `apps/web`: the site and lab, a Next.js static export.

## Develop

```
bun install
bun run dev        # http://localhost:3000/lab/
bun run test       # engine tests
bun run build      # static site in apps/web/out
```

Set `NEXT_PUBLIC_SITE_URL` to the deployed address for the sitemap and share tags (default `https://tessera.shyamsundhar.com`).

## The text format

```
model "Small CNN with a skip"
input 3x32x32
stem = conv 32 k3 -> bn -> relu
y = conv 32 -> bn -> relu -> conv 32 -> bn
join = add(y, stem) -> relu
block "Decoder layer" x4 {
  h = rmsnorm -> attention 4 kv=2
  add(h, @in)
}
```

A chain continues from the line above unless it starts with `@name` or a join such as `add(a, b)`. `name =` names the chain's last node. Inside a block, `@in` is the block's input; `xN` repeats it.

Made by [Shyam Sundhar](https://shyamsundhar.com).
