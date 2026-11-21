"use client";

import { Play, Square } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import type { Model } from "tessera-engine";
import type { DatasetName, Evaluation, MetricComparison, TrainEvent, TrainResult } from "tessera-lab";
import { updateNode } from "@/lib/edit";
import { useLab } from "@/lib/store";
import { cn } from "@/lib/utils";
import { BandChart, Confusion } from "./chart";

const COLOURS = ["#5fb4ff", "#f0569f"];
const DATASET_INFO: Record<DatasetName, { label: string; shape: [number, number, number] }> = {
  mnist: { label: "MNIST digits", shape: [1, 28, 28] },
  "fashion-mnist": { label: "Fashion-MNIST", shape: [1, 28, 28] },
  cifar10: { label: "CIFAR-10 photos", shape: [3, 32, 32] },
};
const METRIC_LABEL: Record<string, string> = { loss: "Test loss", accuracy: "Accuracy", errorRate: "Error rate", macroF1: "Macro F1" };

/** Points a model at a dataset: its input shape, and a final layer with one output per class. */
function adapt(model: Model, dataset: DatasetName): Model {
  const [c, h, w] = DATASET_INFO[dataset].shape;
  let next = model;
  const input = model.nodes.find((node) => node.op === "input");
  if (input) next = updateNode(next, input.id, { args: { shape: `${c}x${h}x${w}` } });
  const last = [...next.nodes].reverse().find((node) => node.op === "linear");
  if (last) next = updateNode(next, last.id, { args: { out: 10 } });
  return next;
}

interface Live {
  /** Per model, per run: test metrics after each epoch. */
  epochs: { accuracy: number; loss: number }[][][];
  status: string;
}

export default function TrainPanel() {
  const slots = useLab((state) => state.slots);
  const edit = useLab((state) => state.edit);
  const setPanel = useLab((state) => state.setPanel);
  const [dataset, setDataset] = useState<DatasetName>("mnist");
  const [train, setTrain] = useState(4000);
  const [epochs, setEpochs] = useState(4);
  const [seeds, setSeeds] = useState(2);
  const [learningRate, setLearningRate] = useState(0.002);
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState<Live | null>(null);
  const [result, setResult] = useState<{ result: TrainResult; comparison: MetricComparison[] | null; classes: string[] } | null>(null);
  const [problems, setProblems] = useState<(string | null)[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const models = slots.filter(Boolean).map((slot) => slot!.model);
  const language = models.some((model) => model.kind === "language");

  // Check each model against the dataset whenever either changes.
  useMemo(() => {
    void import("tessera-lab").then(({ checkCompatibility, trainingProblem, DATASETS }) => {
      setProblems(models.map((model) => trainingProblem(model) ?? checkCompatibility(model, DATASETS[dataset])));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slots, dataset]);

  if (language)
    return (
      <div className="space-y-3 p-4 text-[13px] leading-relaxed">
        <p className="font-medium">Language models train in Colab</p>
        <p className="text-muted">Training even a small language model needs a GPU. The Colab export shrinks it to about 30M parameters with the same wiring and trains it on TinyStories, over several seeds.</p>
        <button type="button" onClick={() => setPanel("code")} className="rounded-lg bg-ink px-3 py-1.5 text-[13px] font-medium text-bg hover:opacity-85">
          Open the Colab export
        </button>
      </div>
    );

  const start = async () => {
    setError(null);
    setResult(null);
    setRunning(true);
    const controller = new AbortController();
    abort.current = controller;
    const state: Live = { epochs: models.map(() => []), status: "Loading the dataset…" };
    setLive({ ...state });
    try {
      const lab = await import("tessera-lab");
      const data = await lab.loadDataset(dataset, { train, test: Math.min(2000, Math.round(train / 4)) }, (progress) => {
        state.status = progress.stage === "download" ? `Downloading ${lab.DATASETS[dataset].label}… ${progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0}%` : `Preparing images… ${progress.loaded}${progress.total ? ` of ${progress.total}` : ""}`;
        setLive({ ...state });
      });
      const seedList = Array.from({ length: seeds }, (_, i) => i);
      const output = await lab.trainCompare(
        models,
        data,
        { epochs, batchSize: 64, learningRate, seed: 0, seeds: seedList },
        (event: TrainEvent) => {
          const name = models[event.modelIndex].name;
          state.status = `${name} · seed ${event.seed} · epoch ${event.epoch}/${epochs} · batch ${event.batch}/${event.batches} · loss ${event.loss.toFixed(3)}`;
          if (event.type === "epoch" && event.evaluation) {
            const runs = (state.epochs[event.modelIndex] ??= []);
            (runs[event.run] ??= []).push({ accuracy: event.evaluation.accuracy, loss: event.evaluation.loss });
          }
          setLive({ ...state, epochs: state.epochs.map((runs) => runs.map((run) => [...run])) });
        },
        controller.signal,
      );
      lab.disposeDataset(data);
      setResult({ result: output, comparison: output.models.length === 2 ? lab.compareRuns(output.models[0], output.models[1]) : null, classes: lab.DATASETS[dataset].classes });
      state.status = output.aborted ? "Stopped." : `Done, on ${output.backend}.`;
      setLive({ ...state });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  const bands = (key: "accuracy" | "loss") =>
    (live?.epochs ?? []).map((runs, index) => {
      const length = Math.max(0, ...runs.map((run) => run.length));
      const mean: number[] = [];
      const std: number[] = [];
      for (let e = 0; e < length; e++) {
        const values = runs.map((run) => run[e]?.[key]).filter((v): v is number => v !== undefined);
        const m = values.reduce((a, b) => a + b, 0) / values.length;
        mean.push(m);
        std.push(values.length > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1)) : 0);
      }
      return { name: models[index]?.name ?? "", colour: COLOURS[index], mean, std };
    });

  const field = "mt-1 w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-[12px] text-ink";
  return (
    <div className="space-y-4 p-4 text-[13px]">
      <div className="grid grid-cols-2 gap-2.5">
        <label className="col-span-2 text-[12px] text-muted">
          Dataset
          <select value={dataset} onChange={(event) => setDataset(event.target.value as DatasetName)} className={field}>
            {Object.entries(DATASET_INFO).map(([id, info]) => (
              <option key={id} value={id}>
                {info.label} ({info.shape.join("×")})
              </option>
            ))}
          </select>
        </label>
        <label className="text-[12px] text-muted">
          Training images
          <select value={train} onChange={(event) => setTrain(Number(event.target.value))} className={field}>
            {[1000, 2000, 4000, 8000, 16000].map((n) => (
              <option key={n} value={n}>
                {n.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[12px] text-muted">
          Epochs
          <input type="number" min={1} max={30} value={epochs} onChange={(event) => setEpochs(Math.max(1, Number(event.target.value) || 1))} className={field} />
        </label>
        <label className="text-[12px] text-muted">
          Seeds
          <input type="number" min={1} max={5} value={seeds} onChange={(event) => setSeeds(Math.max(1, Math.min(5, Number(event.target.value) || 1)))} className={field} />
        </label>
        <label className="text-[12px] text-muted">
          Learning rate
          <input type="number" step={0.0005} min={0.0001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value) || 0.001)} className={field} />
        </label>
      </div>

      {models.map((model, index) =>
        problems[index] ? (
          <div key={index} className="rounded-lg border border-warn/50 bg-warn/10 px-3 py-2 text-[12.5px]">
            <p>
              <span className="font-medium" style={{ color: COLOURS[index] }}>
                {index ? "B" : "A"}
              </span>{" "}
              {problems[index]}
            </p>
            {/(expects|outputs|classes)/i.test(problems[index]!) && (
              <button type="button" onClick={() => edit(index as 0 | 1, (m) => adapt(m, dataset))} className="mt-1.5 rounded-md border border-line px-2 py-0.5 text-[12px] hover:border-muted">
                Adapt it to {DATASET_INFO[dataset].label}
              </button>
            )}
          </div>
        ) : null,
      )}

      <div className="flex items-center gap-2">
        {running ? (
          <button type="button" onClick={() => abort.current?.abort()} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] hover:border-muted">
            <Square className="size-3.5" />
            Stop
          </button>
        ) : (
          <button type="button" disabled={problems.some(Boolean)} onClick={() => void start()} className="flex items-center gap-1.5 rounded-lg bg-ink px-3 py-1.5 text-[13px] font-medium text-bg hover:opacity-85 disabled:opacity-40">
            <Play className="size-3.5" />
            Train {models.length > 1 ? "both" : ""}
          </button>
        )}
        <span className="text-[11.5px] text-muted">Runs in this browser; nothing is uploaded.</span>
      </div>
      {live && <p className="font-mono text-[11.5px] text-muted">{live.status}</p>}
      {error && <p className="rounded-md border border-danger/50 bg-danger/10 px-2.5 py-2 text-[12.5px] text-danger">{error}</p>}

      {live && (
        <div className="space-y-3">
          <div className="flex gap-3 text-[11.5px]">
            {models.map((model, index) => (
              <span key={index} className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ background: COLOURS[index] }} />
                {model.name}
              </span>
            ))}
          </div>
          <BandChart label="Test accuracy (band: ± one standard deviation across seeds)" series={bands("accuracy")} percent />
          <BandChart label="Test loss" series={bands("loss")} />
        </div>
      )}

      {result && <Results result={result.result} comparison={result.comparison} classes={result.classes} />}
    </div>
  );
}

function Results({ result, comparison, classes }: { result: TrainResult; comparison: MetricComparison[] | null; classes: string[] }) {
  const [shown, setShown] = useState(0);
  const model = result.models[shown];
  const finals = model?.runs.map((run) => run.final).filter((f): f is Evaluation => !!f) ?? [];
  const last = finals[finals.length - 1];
  const norms = model?.runs[0]?.epochs[model.runs[0].epochs.length - 1]?.gradientNorms ?? {};
  const topNorms = Object.entries(norms).sort((a, b) => b[1] - a[1]);
  const maxNorm = Math.max(1e-9, ...topNorms.map(([, v]) => v));
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const fmt = (metric: string, v: number) => (metric === "loss" ? v.toFixed(3) : metric === "macroF1" ? v.toFixed(3) : pct(v));

  return (
    <div className="space-y-4">
      <table className="w-full text-[12px]">
        <thead className="text-[11.5px] text-muted">
          <tr>
            <th className="pb-1 text-left font-normal">Final, mean ± std</th>
            {result.models.map((m, i) => (
              <th key={i} className="pb-1 text-right font-normal" style={{ color: COLOURS[i] }}>
                {i ? "B" : "A"}
              </th>
            ))}
            {comparison && <th className="pb-1 text-right font-normal">B vs A</th>}
          </tr>
        </thead>
        <tbody>
          {(["loss", "accuracy", "errorRate", "macroF1"] as const).map((metric) => {
            const row = comparison?.find((c) => c.metric === metric);
            return (
              <tr key={metric} className="border-t border-line">
                <td className="py-1.5 text-muted">{METRIC_LABEL[metric]}</td>
                {result.models.map((m, i) => (
                  <td key={i} className="py-1.5 text-right font-mono">
                    {fmt(metric, m.summary[metric].mean)}
                    {m.summary[metric].n > 1 && <span className="text-muted"> ± {fmt(metric, m.summary[metric].std)}</span>}
                  </td>
                ))}
                {comparison && (
                  <td className={cn("py-1.5 text-right text-[11.5px]", row?.beyondNoise ? (row.better === "b" ? "text-added" : "text-removed") : "text-muted")}>
                    {row?.beyondNoise === null ? "1 seed" : row?.beyondNoise ? `${row.better.toUpperCase()} better` : "within noise"}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {comparison && <p className="text-[11.5px] text-muted">“Better” means the gap between the means is larger than both spreads added together, a rough check rather than a significance test. More seeds make it more trustworthy.</p>}

      {result.models.length > 1 && (
        <div className="flex gap-1">
          {result.models.map((m, i) => (
            <button key={i} type="button" onClick={() => setShown(i)} className={cn("rounded-md px-2.5 py-1 text-[12px] text-muted", shown === i && "bg-surface-3 text-ink")}>
              {i ? "B" : "A"}: {m.name}
            </button>
          ))}
        </div>
      )}
      {last && (
        <>
          <div>
            <p className="mb-1.5 text-[12px] text-muted">Confusion matrix, last seed{last.top5Accuracy !== undefined ? ` · top-5 accuracy ${pct(last.top5Accuracy)}` : ""}</p>
            <Confusion matrix={last.confusion} classes={classes} />
          </div>
          <table className="w-full text-[11.5px]">
            <thead className="text-muted">
              <tr>
                <th className="text-left font-normal">Class</th>
                <th className="text-right font-normal">Precision</th>
                <th className="text-right font-normal">Recall</th>
                <th className="text-right font-normal">F1</th>
              </tr>
            </thead>
            <tbody>
              {last.perClass.map((scores, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="py-0.5">{classes[i]}</td>
                  <td className="py-0.5 text-right font-mono">{scores.precision.toFixed(2)}</td>
                  <td className="py-0.5 text-right font-mono">{scores.recall.toFixed(2)}</td>
                  <td className="py-0.5 text-right font-mono">{scores.f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {topNorms.length > 0 && (
        <div>
          <p className="mb-1.5 text-[12px] text-muted">Which layers were learning: gradient size per layer, last epoch</p>
          <ul className="space-y-1">
            {topNorms.slice(0, 12).map(([id, value]) => (
              <li key={id} className="grid grid-cols-[110px_1fr_52px] items-center gap-2 font-mono text-[11px]">
                <span className="truncate">{id}</span>
                <span className="h-1.5 rounded-full bg-surface-3">
                  <span className="block h-full rounded-full" style={{ width: `${(value / maxNorm) * 100}%`, background: COLOURS[shown] }} />
                </span>
                <span className="text-right text-muted">{value.toExponential(1)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
