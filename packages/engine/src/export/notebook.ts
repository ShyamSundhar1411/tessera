import { shrinkLanguageModel, type ShrinkOptions } from "../shrink";
import type { Model } from "../types";
import { toPyTorch } from "./pytorch";

// A Colab notebook that trains one or two architectures the same way, over
// several random seeds, and reports how they compare: loss and accuracy
// curves with a band for seed-to-seed spread, error rate, top-5 accuracy,
// per-class precision, recall and F1, confusion matrices, and whether the gap
// between the two is larger than the noise.
//
// Image models train on MNIST, Fashion-MNIST or CIFAR-10, picked from the
// input shape. Language models are shrunk to a trainable size, keeping their
// wiring, and trained on TinyStories.

export interface NotebookOptions {
  epochs?: number;
  seeds?: number[];
  batchSize?: number;
  learningRate?: number;
  /** For image models: which dataset to use. */
  dataset?: "auto" | "mnist" | "fashion-mnist" | "cifar10";
  /** For language models: how small to make them and how long to train. */
  shrink?: ShrinkOptions;
  steps?: number;
}

const cell = (type: "markdown" | "code", source: string) => ({
  cell_type: type,
  metadata: {},
  source: source.split("\n").map((line, i, all) => (i < all.length - 1 ? `${line}\n` : line)),
  ...(type === "code" ? { outputs: [], execution_count: null } : {}),
});

const className = (index: number) => `Model${String.fromCharCode(65 + index)}`;

/** Removes the `if __name__ == "__main__"` demo from exported code, so cells only define the class. */
const definitionOnly = (code: string) => code.replace(/\n\nif __name__ == "__main__":[\s\S]*$/, "\n");

function pickDataset(model: Model, choice: NotebookOptions["dataset"]): { name: string; channels: number; size: number } {
  const input = model.nodes.find((node) => node.op === "input");
  const shape = String(input?.args.shape ?? "3x32x32")
    .split(/[x,]/)
    .map(Number);
  const [channels, size] = shape.length === 3 ? [shape[0], shape[1]] : [1, 28];
  const name = choice && choice !== "auto" ? choice : channels === 1 ? "mnist" : "cifar10";
  return { name, channels, size };
}

export function toNotebook(models: Model[], options: NotebookOptions = {}): string {
  const seeds = options.seeds?.length ? options.seeds : [0, 1, 2];
  const language = models.every((model) => model.kind === "language");
  const names = models.map((model) => model.name);
  const cells = [];

  cells.push(
    cell(
      "markdown",
      `# ${models.length > 1 ? `${names[0]} vs ${names[1]}` : names[0]}

Made with **Tessera**. ${
        language
          ? "Both models are shrunk to a trainable size, keeping their wiring, then trained on TinyStories."
          : "Each model trains on the same data with the same settings."
      } Every model runs once per seed (${seeds.join(", ")}), so you can see whether a difference is bigger than run-to-run noise.

Runtime: use a GPU (Runtime → Change runtime type → T4 GPU). It also runs on CPU, slowly.`,
    ),
  );
  cells.push(cell("code", `!pip -q install torch torchvision scikit-learn scipy matplotlib${language ? " datasets transformers" : ""}`));

  const prepared = language ? models.map((model) => shrinkLanguageModel(model, { vocab: 50257, ...options.shrink })) : models;
  prepared.forEach((model, index) => {
    cells.push(cell("markdown", `## ${className(index)}: ${model.name}`));
    cells.push(cell("code", definitionOnly(toPyTorch(model, { className: className(index) }))));
  });

  const modelList = prepared.map((model, index) => `("${model.name.replace(/"/g, "'")}", ${className(index)})`).join(", ");

  if (!language) {
    const { name, channels, size } = pickDataset(prepared[0], options.dataset);
    const datasetClass = { mnist: "MNIST", "fashion-mnist": "FashionMNIST", cifar10: "CIFAR10" }[name] ?? "CIFAR10";
    const native = name === "cifar10" ? { channels: 3, size: 32 } : { channels: 1, size: 28 };
    const transforms = [
      native.size !== size ? `T.Resize(${size})` : "",
      native.channels !== channels ? `T.Grayscale(${channels})` : "",
      "T.ToTensor()",
    ].filter(Boolean);
    cells.push(
      cell(
        "code",
        `import random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, SEEDS = ${options.epochs ?? 5}, ${options.batchSize ?? 128}, ${options.learningRate ?? 1e-3}, ${JSON.stringify(seeds)}
MODELS = [${modelList}]

transform = T.Compose([${transforms.join(", ")}])
train_set = torchvision.datasets.${datasetClass}("data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.${datasetClass}("data", train=False, download=True, transform=transform)
CLASSES = train_set.classes
print(f"{len(train_set)} training and {len(test_set)} test images, {len(CLASSES)} classes, on {device}")`,
      ),
    );
    cells.push(
      cell(
        "code",
        `def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate(model, loader):
    model.eval(); losses, preds, labels, top5 = [], [], [], 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y, reduction="sum").item())
            preds.append(logits.argmax(1).cpu()); labels.append(y.cpu())
            if logits.shape[1] > 5:
                top5 += (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=range(len(CLASSES)), zero_division=0)
    accuracy = (preds == labels).mean()
    return {
        "loss": sum(losses) / len(labels), "accuracy": accuracy, "error_rate": 1 - accuracy,
        "top5_accuracy": top5 / len(labels), "precision": p, "recall": r, "f1": f1, "macro_f1": f1.mean(),
        "confusion": confusion_matrix(labels, preds, labels=range(len(CLASSES))),
    }

def run(name, Model, seed):
    set_seed(seed)
    model = Model().to(device)
    g = torch.Generator().manual_seed(seed)
    train = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True, generator=g, num_workers=2)
    test = torch.utils.data.DataLoader(test_set, batch_size=512, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history = []
    for epoch in range(EPOCHS):
        model.train(); start, total = time.time(), 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(y)
        result = evaluate(model, test)
        result["train_loss"] = total / len(train_set)
        history.append(result)
        print(f"{name} seed {seed} epoch {epoch + 1}: train loss {result['train_loss']:.3f}, test loss {result['loss']:.3f}, "
              f"accuracy {result['accuracy']:.2%}, error {result['error_rate']:.2%}, macro F1 {result['macro_f1']:.3f} ({time.time() - start:.0f}s)")
    return history

results = {name: [run(name, Model, seed) for seed in SEEDS] for name, Model in MODELS}`,
      ),
    );
    cells.push(
      cell(
        "code",
        `import matplotlib.pyplot as plt
from scipy import stats

METRICS = ["loss", "accuracy", "error_rate", "top5_accuracy", "macro_f1"]
final = {name: {m: np.array([h[-1][m] for h in runs]) for m in METRICS} for name, runs in results.items()}
print(f"{'metric':<15}" + "".join(f"{name[:28]:>32}" for name in final))
for m in METRICS:
    print(f"{m:<15}" + "".join(f"{v[m].mean():>20.4f} ± {v[m].std():<9.4f}" for v in final.values()))

if len(final) == 2 and len(SEEDS) > 1:
    (a, va), (b, vb) = final.items()
    print("\\nIs the gap bigger than seed noise? (Welch's t-test across seeds)")
    for m in METRICS:
        t, p = stats.ttest_ind(va[m], vb[m], equal_var=False)
        verdict = "a real difference" if p < 0.05 else "within noise"
        print(f"  {m:<15} {b[:20]} minus {a[:20]}: {vb[m].mean() - va[m].mean():+.4f}  p={p:.3f}  {verdict}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for name, runs in results.items():
    for ax, m in zip(axes, ["loss", "accuracy", "macro_f1"]):
        curves = np.array([[h[e][m] for e in range(EPOCHS)] for h in runs])
        mean, std = curves.mean(0), curves.std(0)
        epochs = np.arange(1, EPOCHS + 1)
        ax.plot(epochs, mean, label=name); ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"test {m} (band: ± one std across seeds)"); ax.set_xlabel("epoch")
axes[0].legend(); plt.show()`,
      ),
    );
    cells.push(
      cell(
        "code",
        `fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
axes = np.atleast_1d(axes)
for ax, (name, runs) in zip(axes, results.items()):
    matrix = runs[0][-1]["confusion"]
    ax.imshow(matrix, cmap="magma"); ax.set_title(f"{name[:30]}: confusion matrix, seed {SEEDS[0]}")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, fontsize=8); ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=6, color="white" if matrix[i, j] < matrix.max() / 2 else "black")
plt.tight_layout(); plt.show()

for name, runs in results.items():
    last = runs[0][-1]
    print(f"\\n{name}: per class (seed {SEEDS[0]})")
    for c, label in enumerate(CLASSES):
        print(f"  {label:<14} precision {last['precision'][c]:.3f}  recall {last['recall'][c]:.3f}  F1 {last['f1'][c]:.3f}")`,
      ),
    );
  } else {
    cells.push(
      cell(
        "code",
        `import math, random, time
import numpy as np
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, BATCH, LR, SEEDS, SEQ = ${options.steps ?? 1500}, ${options.batchSize ?? 32}, ${options.learningRate ?? 6e-4}, ${JSON.stringify(seeds)}, ${prepared[0].seqLen ?? 256}
MODELS = [${modelList}]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
stories = load_dataset("roneneldan/TinyStories", split="train[:2%]")
ids = torch.tensor(tokenizer("\\n".join(stories["text"]))["input_ids"])
split = int(len(ids) * 0.95)
train_ids, val_ids = ids[:split], ids[split:]
print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens, on {device}")`,
      ),
    );
    cells.push(
      cell(
        "code",
        `def batch(data, g):
    starts = torch.randint(0, len(data) - SEQ - 1, (BATCH,), generator=g)
    x = torch.stack([data[s : s + SEQ] for s in starts])
    y = torch.stack([data[s + 1 : s + SEQ + 1] for s in starts])
    return x.to(device), y.to(device)

def evaluate(model, g, batches=20):
    model.eval(); losses, correct, total = [], 0, 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = batch(val_ids, g)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    model.train(); loss = float(np.mean(losses))
    return {"loss": loss, "perplexity": math.exp(loss), "token_accuracy": correct / total, "token_error_rate": 1 - correct / total}

def run(name, Model, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Model().to(device); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    history, start = [], time.time()
    for step in range(1, STEPS + 1):
        x, y = batch(train_ids, g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % max(1, STEPS // 15) == 0 or step == STEPS:
            result = evaluate(model, torch.Generator().manual_seed(1234)); result["step"] = step; result["train_loss"] = loss.item()
            history.append(result)
            print(f"{name} seed {seed} step {step}: val loss {result['loss']:.3f}, perplexity {result['perplexity']:.1f}, "
                  f"token accuracy {result['token_accuracy']:.2%} ({time.time() - start:.0f}s)")
    return history

for name, Model in MODELS:
    print(f"{name}: {sum(p.numel() for p in Model().parameters()):,} parameters")
results = {name: [run(name, Model, seed) for seed in SEEDS] for name, Model in MODELS}`,
      ),
    );
    cells.push(
      cell(
        "code",
        `import matplotlib.pyplot as plt
from scipy import stats

METRICS = ["loss", "perplexity", "token_accuracy", "token_error_rate"]
final = {name: {m: np.array([h[-1][m] for h in runs]) for m in METRICS} for name, runs in results.items()}
print(f"{'metric':<18}" + "".join(f"{name[:28]:>32}" for name in final))
for m in METRICS:
    print(f"{m:<18}" + "".join(f"{v[m].mean():>20.4f} ± {v[m].std():<9.4f}" for v in final.values()))

if len(final) == 2 and len(SEEDS) > 1:
    (a, va), (b, vb) = final.items()
    print("\\nIs the gap bigger than seed noise? (Welch's t-test across seeds)")
    for m in METRICS:
        t, p = stats.ttest_ind(va[m], vb[m], equal_var=False)
        print(f"  {m:<18} {vb[m].mean() - va[m].mean():+.4f}  p={p:.3f}  {'a real difference' if p < 0.05 else 'within noise'}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, runs in results.items():
    steps = [h["step"] for h in runs[0]]
    for ax, m in zip(axes, ["loss", "token_accuracy"]):
        curves = np.array([[h[m] for h in run] for run in runs])
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(steps, mean, label=name); ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"validation {m} (band: ± one std across seeds)"); ax.set_xlabel("step")
axes[0].legend(); plt.show()`,
      ),
    );
  }

  return JSON.stringify(
    {
      cells,
      metadata: { kernelspec: { name: "python3", display_name: "Python 3" }, language_info: { name: "python" }, accelerator: "GPU", colab: { provenance: [] } },
      nbformat: 4,
      nbformat_minor: 5,
    },
    null,
    1,
  );
}
