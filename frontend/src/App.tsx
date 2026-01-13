import { useEffect, useMemo, useState } from "react";
import { Button } from "./components/ui/button";

interface DatasetInfo {
  size: number;
  classes: string[];
}

interface AttackResult {
  attack: string;
  success: boolean | null;
  pred_before: string | null;
  pred_after: string | null;
  linf: number | null;
  l2: number | null;
  time_ms: number | null;
  status: string;
}

interface JobStatus {
  id: string;
  status: string;
  progress: number;
  results: AttackResult[];
  index: number;
}

interface InferResult {
  index: number;
  label: number | null;
  latency_ms: number;
  top_k: { class: string; index: number; score: number }[];
}

export default function App() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [index, setIndex] = useState<number>(0);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [advUrl, setAdvUrl] = useState<string>("");
  const [diffUrl, setDiffUrl] = useState<string>("");
  const [zoom, setZoom] = useState<number>(1);
  const [grid, setGrid] = useState<boolean>(false);
  const [inferResult, setInferResult] = useState<InferResult | null>(null);
  const [attacks, setAttacks] = useState<string[]>([]);
  const [selectedAttacks, setSelectedAttacks] = useState<Record<string, boolean>>({});
  const [job, setJob] = useState<JobStatus | null>(null);
  const [selectedAttack, setSelectedAttack] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/dataset/info")
      .then((res) => res.json())
      .then(setDatasetInfo);
    fetch("/api/v1/attacks")
      .then((res) => res.json())
      .then((data) => {
        setAttacks(data.attacks || []);
        const initial: Record<string, boolean> = {};
        (data.attacks || []).forEach((name: string) => (initial[name] = true));
        setSelectedAttacks(initial);
      });
  }, []);

  useEffect(() => {
    setPreviewUrl(`/api/v1/images/${index}?format=png`);
    setAdvUrl("");
    setDiffUrl("");
  }, [index]);

  const scaleStyle = useMemo(
    () => ({
      width: `${96 * zoom}px`,
      height: `${96 * zoom}px`,
    }),
    [zoom]
  );

  const handleInfer = async () => {
    const res = await fetch("/api/v1/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index, top_k: 5 }),
    });
    const data = await res.json();
    setInferResult(data);
  };

  const handleRunAttacks = async () => {
    const attacksList = attacks
      .filter((name) => selectedAttacks[name])
      .map((name) => ({ name }));
    const res = await fetch("/api/v1/jobs/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index, attacks: attacksList }),
    });
    const data = await res.json();
    setJob({ id: data.job_id, status: "queued", progress: 0, results: [], index });
  };

  useEffect(() => {
    if (!job) return;
    const timer = setInterval(async () => {
      const res = await fetch(`/api/v1/jobs/${job.id}`);
      const data = await res.json();
      setJob({ ...data, id: job.id });
      if (data.results && data.results.length && !selectedAttack) {
        setSelectedAttack(data.results[0].attack);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [job, selectedAttack]);

  useEffect(() => {
    if (!job || !selectedAttack) return;
    setAdvUrl(`/api/v1/jobs/${job.id}/artifacts/${selectedAttack}?type=adv&format=png`);
    setDiffUrl(`/api/v1/jobs/${job.id}/artifacts/${selectedAttack}?type=diff&format=png&amplify=10`);
  }, [job, selectedAttack]);

  return (
    <div className="min-h-screen px-6 py-6">
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[320px_1fr]">
        <aside className="space-y-6">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-lg font-semibold">Dataset</h2>
            <p className="text-sm text-slate-400">Размер: {datasetInfo?.size ?? "-"}</p>
            <div className="mt-4 flex items-center gap-2">
              <Button variant="outline" onClick={() => setIndex((prev) => Math.max(prev - 1, 0))}>
                Prev
              </Button>
              <input
                type="number"
                className="w-full rounded-md border border-slate-700 bg-slate-950 p-2 text-slate-100"
                value={index}
                onChange={(e) => setIndex(Number(e.target.value))}
              />
              <Button
                variant="outline"
                onClick={() =>
                  setIndex((prev) =>
                    datasetInfo ? Math.min(prev + 1, datasetInfo.size - 1) : prev + 1
                  )
                }
              >
                Next
              </Button>
            </div>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-lg font-semibold">Inference</h2>
            <Button className="mt-3" variant="primary" onClick={handleInfer}>
              Run inference
            </Button>
            {inferResult && (
              <div className="mt-4 space-y-2 text-sm">
                <div>Latency: {inferResult.latency_ms.toFixed(1)} ms</div>
                <div>True label: {inferResult.label ?? "-"}</div>
                <ul className="space-y-1 text-slate-300">
                  {inferResult.top_k.map((item) => (
                    <li key={item.index}>
                      {item.class} ({item.score.toFixed(3)})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-lg font-semibold">Attacks</h2>
            <div className="mt-3 space-y-2 text-sm">
              {attacks.map((name) => (
                <label key={name} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedAttacks[name] ?? false}
                    onChange={(e) =>
                      setSelectedAttacks((prev) => ({ ...prev, [name]: e.target.checked }))
                    }
                  />
                  <span>{name}</span>
                </label>
              ))}
            </div>
            <Button className="mt-4 w-full" variant="primary" onClick={handleRunAttacks}>
              Run attacks
            </Button>
            {job && (
              <div className="mt-4 text-sm text-slate-300">
                <div>Status: {job.status}</div>
                <div>Progress: {job.progress}%</div>
              </div>
            )}
          </section>
        </aside>

        <main className="space-y-6">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Preview (96×96)</h2>
              <div className="flex items-center gap-2 text-sm">
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={grid} onChange={(e) => setGrid(e.target.checked)} />
                  Pixel grid
                </label>
                <select
                  className="rounded-md border border-slate-700 bg-slate-950 p-1 text-slate-100"
                  value={zoom}
                  onChange={(e) => setZoom(Number(e.target.value))}
                >
                  <option value={1}>×1</option>
                  <option value={4}>×4</option>
                  <option value={8}>×8</option>
                </select>
              </div>
            </div>
            <div className="mt-4 grid gap-6 md:grid-cols-3">
              {[{ label: "Original", url: previewUrl }, { label: "Adversarial", url: advUrl }, { label: "Diff", url: diffUrl }].map(
                (item) => (
                  <div key={item.label}>
                    <div className="text-xs uppercase text-slate-400">{item.label}</div>
                    <div
                      className={`mt-2 flex items-center justify-center rounded-lg border border-slate-800 bg-slate-950 p-2 ${grid ? "pixel-grid" : ""}`}
                    >
                      {item.url ? (
                        <img
                          src={item.url}
                          alt={item.label}
                          style={scaleStyle}
                          className="pixelated"
                        />
                      ) : (
                        <div className="text-xs text-slate-500">No image</div>
                      )}
                    </div>
                  </div>
                )
              )}
            </div>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-lg font-semibold">Attack results</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full text-sm text-slate-200">
                <thead>
                  <tr className="border-b border-slate-700 text-left">
                    <th className="py-2 pr-3">Attack</th>
                    <th className="py-2 pr-3">Success</th>
                    <th className="py-2 pr-3">Pred (before)</th>
                    <th className="py-2 pr-3">Pred (after)</th>
                    <th className="py-2 pr-3">Linf</th>
                    <th className="py-2 pr-3">L2</th>
                    <th className="py-2 pr-3">Time</th>
                    <th className="py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(job?.results || []).map((row) => (
                    <tr
                      key={row.attack}
                      className={`border-b border-slate-800 hover:bg-slate-800/40 ${
                        selectedAttack === row.attack ? "bg-slate-800/60" : ""
                      }`}
                      onClick={() => setSelectedAttack(row.attack)}
                    >
                      <td className="py-2 pr-3 font-medium">{row.attack}</td>
                      <td className="py-2 pr-3">{row.success ? "Да" : row.success === false ? "Нет" : "-"}</td>
                      <td className="py-2 pr-3">{row.pred_before ?? "-"}</td>
                      <td className="py-2 pr-3">{row.pred_after ?? "-"}</td>
                      <td className="py-2 pr-3">{row.linf ?? "-"}</td>
                      <td className="py-2 pr-3">{row.l2 ?? "-"}</td>
                      <td className="py-2 pr-3">{row.time_ms ?? "-"}</td>
                      <td className="py-2">{row.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
