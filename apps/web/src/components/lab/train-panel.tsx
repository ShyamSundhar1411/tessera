"use client";

import { Play, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { formatCount, type Model } from "tessera-engine";
import type { AnyTrainEvent, AnyTrainResult, DatasetName, Evaluation, MetricComparison, TaskSpec } from "tessera-lab";
import { useLab } from "@/lib/store";
import { useTrained } from "@/lib/trained";
import { cn } from "@/lib/utils";
import { BandChart, Confusion } from "./chart";

const COLOURS = ["#5fb4ff", "#f0569f"];
const IMAGE_DATASETS: { id: DatasetName; label: string }[] = [
  { id: "mnist", label: "MNIST digits" },
  { id: "fashion-mnist", label: "Fashion-MNIST" },
  { id: "cifar10", label: "CIFAR-10 photos" },
];
const METRIC_LABEL: Record<string, string> = { loss: "Test loss", accuracy: "Accuracy", errorRate: "Error rate", macroF1: "Macro F1", perplexity: "Perplexity" };

interface Plan {
  task: TaskSpec;
  prepared: { description: string; params: number; originalParams: number }[];
}

interface Live {
  epochs: { accuracy: number; loss: number; perplexity?: number }[][][];
  status: string;
}

export default function TrainPanel() {
  const slots = useLab((state) => state.slots);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [reason, setReason] = useState<string | null>(null);
  const [dataset, setDataset] = useState<DatasetName | null>(null);
  const [examples, setExamples] = useState(2000);
  const [epochs, setEpochs] = useState(4);
  const [seeds, setSeeds] = useState(2);
  const [learningRate, setLearningRate] = useState(0.002);
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState<Live | null>(null);
  const [result, setResult] = useState<{ result: AnyTrainResult; comparison: MetricComparison[] | null; classes: string[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const models: Model[] = slots.filter(Boolean).map((slot) => slot!.model);
  const slotOf = ([0, 1] as const).filter((index) => slots[index]);

  // Work out what these models can learn, and how they would be shrunk to train here.
  useEffect(() => {
    let live = true;
    void import("tessera-lab").then(({ planCompare }) => {
      if (!live) return;
      const outcome = planCompare(models, dataset ? { imageDataset: dataset } : undefined);
      if (outcome.ok) {
        setPlan({ task: outcome.task, prepared: outcome.prepared.map((item) => ({ description: item.description, params: item.params, originalParams: item.originalParams })) });
        setReason(null);
      } else {
        setPlan(null);
        setReason(outcome.reason);
      }
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slots, dataset]);

  const imageTask = plan?.task.kind === "image-classification";

  const start = async () => {
    if (!plan) return;
    setError(null);
    setResult(null);
    setRunning(true);
    const controller = new AbortController();
    abort.current = controller;
    const state: Live = { epochs: models.map(() => []), status: "Preparing the data…" };
    setLive({ ...state });
    try {
      const lab = await import("tessera-lab");
      let imageDataset;
      let classes: string[] = [];
      if (imageTask) {
        const name = plan.task.imageDataset ?? "mnist";
        imageDataset = await lab.loadDataset(name, { train: examples, test: Math.min(2000, Math.round(examples / 4)) }, (progress) => {
          state.status = progress.stage === "download" ? `Downloading ${lab.DATASETS[name].label}… ${progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0}%` : `Preparing images… ${progress.loaded}${progress.total ? ` of ${progress.total}` : ""}`;
          setLive({ ...state });
        });
        classes = lab.DATASETS[name].classes;
      } else if (plan.task.kind === "sequence-classification") classes = Array.from({ length: plan.task.classes }, (_, i) => `class ${i}`);
      const output = await lab.trainCompareAny(
        models,
        {
          epochs,
          batchSize: imageTask ? 64 : 32,
          learningRate,
          seed: 0,
          seeds: Array.from({ length: seeds }, (_, i) => i),
          imageDataset,
          trainExamples: examples,
          testExamples: Math.max(128, Math.round(examples / 8)),
          ...(dataset ? { prepare: { imageDataset: dataset } } : {}),
          // The first seed's weights stay alive for the trace view; the rest are freed.
          keepTrained: ({ modelIndex, run, graph, prepared }) => {
            if (run === 0) useTrained.getState().keep(slotOf[modelIndex], { source: models[modelIndex], graph, prepared, kind: "trained" });
            else graph.dispose();
          },
        },
        (event: AnyTrainEvent) => {
          state.status = `${models[event.modelIndex].name} · seed ${event.seed} · epoch ${event.epoch}/${epochs} · batch ${event.batch}/${event.batches} · loss ${event.loss.toFixed(3)}`;
          if (event.type === "epoch" && event.evaluation) {
            const runs = (state.epochs[event.modelIndex] ??= []);
            (runs[event.run] ??= []).push({ accuracy: event.evaluation.accuracy, loss: event.evaluation.loss, perplexity: event.evaluation.perplexity });
          }
          setLive({ ...state, epochs: state.epochs.map((runs) => runs.map((run) => [...run])) });
        },
        controller.signal,
      );
      if (imageDataset) lab.disposeDataset(imageDataset);
      setResult({ result: output, comparison: output.models.length === 2 ? lab.compareRuns(output.models[0], output.models[1]) : null, classes });
      state.status = output.aborted ? "Stopped." : `Done, on ${output.backend}.`;
      setLive({ ...state });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  const bands = (key: "accuracy" | "loss" | "perplexity") =>
    (live?.epochs ?? []).map((runs, index) => {
      const length = Math.max(0, ...runs.map((run) => run.length));
      const mean: number[] = [];
      const std: number[] = [];
      for (let e = 0; e < length; e++) {
        const values = runs.map((run) => run[e]?.[key]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
        if (!values.length) continue;
        const m = values.reduce((a, b) => a + b, 0) / values.length;
        mean.push(m);
        std.push(values.length > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1)) : 0);
      }
      return { name: models[index]?.name ?? "", colour: COLOURS[index], mean, std };
    });

  const field = "mt-1 w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-[12px] text-ink";
  return (
    <div className="space-y-4 p-4 text-[13px]">
      {plan ? (
        <div className="rounded-lg border border-line bg-surface-2 p-3">
          <p className="text-[12px] text-muted">What they will learn</p>
          <p className="mt-0.5 font-medium">{plan.task.label}</p>
          <p className="mt-0.5 text-[11.5px] text-muted">Scored by {plan.task.accuracyLabel}. Data: {plan.task.source}.</p>
          <ul className="mt-2 space-y-1.5">
            {plan.prepared.map((item, index) => (
              <li key={index} className="text-[12px] leading-relaxed">
                <span className="font-semibold" style={{ color: COLOURS[index] }}>
                  {index ? "B" : "A"}
                </span>{" "}
                {item.params !== item.originalParams ? (
                  <>
                    trains at <span className="font-mono">{formatCount(item.params)}</span> parameters, shrunk from <span className="font-mono">{formatCount(item.originalParams)}</span> with the same wiring.
                  </>
                ) : (
                  <>
                    trains as is, <span className="font-mono">{formatCount(item.params)}</span> parameters.
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        reason && <p className="rounded-lg border border-warn/50 bg-warn/10 px-3 py-2 text-[12.5px]">{reason}</p>
      )}

      <div className="grid grid-cols-2 gap-2.5">
        {imageTask && (
          <label className="col-span-2 text-[12px] text-muted">
            Dataset
            <select value={dataset ?? plan?.task.imageDataset ?? "mnist"} onChange={(event) => setDataset(event.target.value as DatasetName)} className={field}>
              {IMAGE_DATASETS.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="text-[12px] text-muted">
          {imageTask ? "Training images" : "Training examples"}
          <select value={examples} onChange={(event) => setExamples(Number(event.target.value))} className={field}>
            {[500, 1000, 2000, 4000, 8000].map((n) => (
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

      <div className="flex items-center gap-2">
        {running ? (
          <button type="button" onClick={() => abort.current?.abort()} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] hover:border-muted">
            <Square className="size-3.5" />
            Stop
          </button>
        ) : (
          <button type="button" disabled={!plan} onClick={() => void start()} className="flex items-center gap-1.5 rounded-lg bg-ink px-3 py-1.5 text-[13px] font-medium text-bg hover:opacity-85 disabled:opacity-40">
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
          <BandChart label={`${plan?.task.accuracyLabel ?? "Accuracy"} (band: ± one standard deviation across seeds)`} series={bands("accuracy")} percent />
          {plan?.task.tokenLevel ? <BandChart label="Perplexity (lower is better)" series={bands("perplexity")} /> : <BandChart label="Test loss" series={bands("loss")} />}
        </div>
      )}

      {result && <Results result={result.result} comparison={result.comparison} classes={result.classes} tokenLevel={!!plan?.task.tokenLevel} />}
    </div>
  );
}

function Results({ result, comparison, classes, tokenLevel }: { result: AnyTrainResult; comparison: MetricComparison[] | null; classes: string[]; tokenLevel: boolean }) {
  const [shown, setShown] = useState(0);
  const model = result.models[shown];
  const finals = model?.runs.map((run) => run.final).filter((f): f is Evaluation => !!f) ?? [];
  const last = finals[finals.length - 1];
  const norms = model?.runs[0]?.epochs[model.runs[0].epochs.length - 1]?.gradientNorms ?? {};
  const topNorms = Object.entries(norms).sort((a, b) => b[1] - a[1]);
  const maxNorm = Math.max(1e-9, ...topNorms.map(([, v]) => v));
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const fmt = (metric: string, v: number) => (metric === "loss" || metric === "macroF1" ? v.toFixed(3) : metric === "perplexity" ? v.toFixed(2) : pct(v));
  const metrics = (tokenLevel ? ["loss", "perplexity", "accuracy", "errorRate"] : ["loss", "accuracy", "errorRate", "macroF1"]) as string[];

  return (
    <div className="space-y-4">
      <table className="w-full text-[12px]">
        <thead className="text-[11.5px] text-muted">
          <tr>
            <th className="pb-1 text-left font-normal">Final, mean ± std</th>
            {result.models.map((_, i) => (
              <th key={i} className="pb-1 text-right font-normal" style={{ color: COLOURS[i] }}>
                {i ? "B" : "A"}
              </th>
            ))}
            {comparison && <th className="pb-1 text-right font-normal">B vs A</th>}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => {
            const row = comparison?.find((c) => c.metric === metric);
            return (
              <tr key={metric} className="border-t border-line">
                <td className="py-1.5 text-muted">{METRIC_LABEL[metric]}</td>
                {result.models.map((m, i) => {
                  const stats = (m.summary as Record<string, { mean: number; std: number; n: number } | undefined>)[metric];
                  return (
                    <td key={i} className="py-1.5 text-right font-mono">
                      {stats ? fmt(metric, stats.mean) : "-"}
                      {stats && stats.n > 1 && <span className="text-muted"> ± {fmt(metric, stats.std)}</span>}
                    </td>
                  );
                })}
                {comparison && (
                  <td className={cn("py-1.5 text-right text-[11.5px]", row?.beyondNoise ? (row.better === "b" ? "text-added" : "text-removed") : "text-muted")}>
                    {!row ? "" : row.beyondNoise === null ? "1 seed" : row.beyondNoise ? `${row.better.toUpperCase()} better` : "within noise"}
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
      {model?.description && <p className="text-[11.5px] leading-relaxed text-muted">{model.description}</p>}
      {last && !tokenLevel && classes.length > 0 && classes.length <= 20 && (
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
      {last?.sequenceAccuracy !== undefined && <p className="text-[12px]">Whole sequences exactly right: <span className="font-mono">{pct(last.sequenceAccuracy)}</span></p>}
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
