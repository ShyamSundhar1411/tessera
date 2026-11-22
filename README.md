# Tessera

See inside any neural network, then rewire it. Load an architecture, from LeNet to DeepSeek-V3, follow your own input through every layer with the maths written out, run real published weights, change its connections, train both versions and export the PyTorch code. Everything runs in the browser.

## The lab

The lab has six modes, picked from the rail on the left. A and B stay loaded across all of them, and every mode shows them side by side.

- **Architecture.** A 2D graph where each layer carries a drawing of what it does (feature maps under a kernel, a fully connected fan, a causal attention matrix, the activation curve), and blocks show their layers, residuals and repeats. Or a 3D view of the tensors themselves. Drag from the dot under one layer onto another to connect them: into a layer that already has an input, that adds a skip (an add when shapes match, a concat when only channels differ). In 3D, click a layer then shift-click another. Inspect and edit any layer, compare A and B, or edit the text form.
- **Trace.** Type text, draw a digit or drop a photo, and follow it through every layer. Each card shows the layer's equation, the same equation with this model's sizes, a plain explanation, and the values it produced: feature maps, token-by-feature heatmaps, attention for every head with token labels, histograms and predictions. Play or step through the pass while the 3D view beside it glides to the current layer and paints its real activations.
- **Maths.** From the wiring alone: longest and shortest gradient paths through skip connections, receptive fields, compute by kind of layer, and FLOPs and key/value cache against context length.
- **Train.** A small version of each model learns a task picked from its wiring (next character for decoders, masked characters for encoders, sequence reversal for encoder-decoders, MNIST or CIFAR-10 for image models) on the GPU with TensorFlow.js, over several seeds, with accuracy, perplexity, error rate, per-class F1, confusion matrices and whether the gap beats seed-to-seed noise.
- **Run.** Runs the model on the canvas: published ImageNet weights for the catalog's image models, OpenAI's GPT-2 writing text token by token (hover a token for the alternatives it weighed), or the weights from the Train tab.
- **Export.** PyTorch, the text format, JSON, or a Colab notebook that trains A and B over several seeds on a free GPU.

## Models

The catalog: decoder LLMs (Llama 3.1, Mistral, Mixtral, Qwen 2.5 and 3, DeepSeek-V3, Gemma 2 and 3, Phi-3, SmolLM2, GPT-2), the original encoder-decoder Transformer from *Attention Is All You Need*, encoders (BERT, RoBERTa, DistilBERT), image models (ViT-B/16, AlexNet, VGG16, GoogLeNet, ResNet-18 and 50, DenseNet-121, SqueezeNet, U-Net, MobileNetV2, EfficientNet-B0) and small classics (LeNet-5, an MLP, small CNNs). The load dialog opens with the classic papers and has a timeline of all of them, 1998 to 2025. Or load any open model's `config.json` from Hugging Face (decoders and BERT-style encoders), an `.onnx` file, or start blank.

Real weights that load into the graph itself, so the trace shows what the trained model does: GPT-2 small (`openai-community/gpt2`, 548 MB) and torchvision's ResNet-18 and ResNet-50 (`timm/resnet*.tv_in1k`). They download once and the browser keeps them.

## Correctness

- Every catalog model matches its published parameter count (image models to the parameter, language models within half a percent), and every exported PyTorch module builds with exactly the engine's count.
- GPT-2 in the browser matches a PyTorch reference: same top five next tokens, scores within 0.01 (the gap is exact versus tanh GELU). The tokenizer gives GPT-2's ids.
- ResNet-18 and 50 match torchvision's logits on a test photo to three decimals.
- Receptive fields match published values (ResNet-50 427 px, VGG16 212 px, AlexNet 195 px).

## Layout

- `packages/engine`: the graph format, operations with shape, parameter and compute rules, analysis, the catalog, Hugging Face and ONNX import, the text format, diff, shrink, PyTorch and notebook export. `bun test` runs its tests.
- `packages/lab`: the TensorFlow.js executor for any graph, tracing, browser training, real weights (safetensors, GPT-2 BPE) and pretrained inference (ONNX Runtime Web).
- `apps/web`: the site and lab, a Next.js static export.

## Develop

```
bun install
bun run dev        # http://localhost:3000/lab/
bun run test       # engine tests (cd packages/lab && bun test for the lab)
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
