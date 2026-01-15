import { useEffect, useMemo, useRef, useState } from "react";
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

type PageKey = "attacks" | "defenses";

const MOCK_INFER: InferResult = {
  index: 1,
  label: 5,
  latency_ms: 176.9,
  top_k: [
    { class: "dog", index: 5, score: 0.468 },
    { class: "horse", index: 6, score: 0.204 },
    { class: "fox", index: 277, score: 0.092 },
    { class: "wolf", index: 269, score: 0.081 },
    { class: "cat", index: 281, score: 0.047 },
  ],
};

const DEFENSE_ROWS = [
  {
    attack: "FGSM",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=5",
    noise: "σ=0.60",
    attempts: 64,
    degradation: "PSNR 11.2 dB",
    status: "Восстановлено",
  },
  {
    attack: "BIM",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=8",
    noise: "σ=0.40",
    attempts: 48,
    degradation: "PSNR 13.8 dB",
    status: "Восстановлено",
  },
  {
    attack: "PGD",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "3 (bear)",
    jpeg: "Q=4",
    noise: "σ=0.75",
    attempts: 64,
    degradation: "PSNR 9.6 dB",
    status: "Не восстановлено",
  },
  {
    attack: "DeepFool",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=6",
    noise: "σ=0.50",
    attempts: 32,
    degradation: "PSNR 12.4 dB",
    status: "Восстановлено",
  },
  {
    attack: "C&W",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "6 (horse)",
    jpeg: "Q=3",
    noise: "σ=0.80",
    attempts: 80,
    degradation: "PSNR 8.1 dB",
    status: "Не восстановлено",
  },
  {
    attack: "AutoAttack",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=7",
    noise: "σ=0.45",
    attempts: 56,
    degradation: "PSNR 14.1 dB",
    status: "Восстановлено",
  },
];

const PREVIEW_VARIANTS: Record<string, string> = {
  original: "bg-gradient-to-br from-slate-500/50 via-slate-900 to-slate-950",
  adversarial: "bg-gradient-to-br from-rose-500/40 via-slate-900 to-slate-950",
  defended: "bg-gradient-to-br from-emerald-500/40 via-slate-900 to-slate-950",
  diff: "bg-gradient-to-br from-sky-500/40 via-slate-900 to-slate-950",
};

export default function App() {
  const [page, setPage] = useState<PageKey>("attacks");
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [index, setIndex] = useState<number>(1);
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
  const [metaError, setMetaError] = useState<string | null>(null);
  const [defenseMode, setDefenseMode] = useState<string>("restore_clean");
  const [defenseStack, setDefenseStack] = useState<string>("jpeg");
  const [defenseOrder, setDefenseOrder] = useState<string>("jpeg_noise");
  const [defenseSeed, setDefenseSeed] = useState<number>(123);
  const [defenseStatus, setDefenseStatus] = useState<string>("idle");
  const [defenseProgress, setDefenseProgress] = useState<number>(0);
  const defenseTimer = useRef<number | null>(null);

  const datasetSize = datasetInfo?.size ?? 5000;
  const apiIndex = Math.max(index - 1, 0);

  const scaleStyle = useMemo(
    () => ({
      width: `${96 * zoom}px`,
      height: `${96 * zoom}px`,
    }),
    [zoom]
  );

  const loadMetadata = async (): Promise<boolean> => {
    try {
      const [datasetRes, attacksRes] = await Promise.all([
        fetch("/api/v1/dataset/info"),
        fetch("/api/v1/attacks"),
      ]);
      if (!datasetRes.ok || !attacksRes.ok) {
        throw new Error("Metadata fetch failed");
      }
      const dataset = await datasetRes.json();
      const attacksData = await attacksRes.json();
      setDatasetInfo(dataset);
      setAttacks(attacksData.attacks || []);
      const initial: Record<string, boolean> = {};
      (attacksData.attacks || []).forEach((name: string) => (initial[name] = true));
      setSelectedAttacks(initial);
      setMetaError(null);
      return true;
    } catch {
      setMetaError("Backend is not ready. Retrying...");
      return false;
    }
  };

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      if (cancelled) return;
      const ok = await loadMetadata();
      if (!ok && !cancelled) {
        timer = setTimeout(tick, 2000);
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    setPreviewUrl(`/api/v1/images/${apiIndex}?format=png`);
    setAdvUrl("");
    setDiffUrl("");
  }, [apiIndex]);

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

  useEffect(() => {
    if (defenseTimer.current) {
      window.clearTimeout(defenseTimer.current);
      defenseTimer.current = null;
    }
    return () => {
      if (defenseTimer.current) {
        window.clearTimeout(defenseTimer.current);
      }
    };
  }, []);

  const clampIndex = (value: number) => Math.min(Math.max(value, 1), datasetSize);
  const safeNumber = (value: number, fallback: number) =>
    Number.isNaN(value) ? fallback : value;

  const handleInfer = async () => {
    setInferResult({ ...MOCK_INFER, index });
  };

  const handleRunAttacks = async () => {
    const attacksList = attacks
      .filter((name) => selectedAttacks[name])
      .map((name) => ({ name }));
    const res = await fetch("/api/v1/jobs/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: apiIndex, attacks: attacksList }),
    });
    const data = await res.json();
    setJob({ id: data.job_id, status: "queued", progress: 0, results: [], index: apiIndex });
  };

  const handleRunDefenses = () => {
    if (defenseStatus === "running defenses") return;
    setDefenseStatus("running defenses");
    setDefenseProgress(0);
    let progress = 0;
    const step = () => {
      progress += 10;
      setDefenseProgress(progress);
      if (progress < 100) {
        defenseTimer.current = window.setTimeout(step, 60 + Math.random() * 40);
      } else {
        setDefenseStatus("done");
      }
    };
    defenseTimer.current = window.setTimeout(step, 60 + Math.random() * 40);
  };

  const attackPreviewItems = [
    { label: "ORIGINAL", subtitle: "5 (dog) (p=0.468)", url: previewUrl, variant: "original" },
    { label: "ADVERSARIAL", subtitle: "6 (horse) (p=1.000)", url: advUrl, variant: "adversarial" },
    { label: "DIFF", subtitle: "|def - adv|", url: diffUrl, variant: "diff" },
  ];

  const defensePreviewItems = [
    { label: "ORIGINAL", subtitle: "5 (dog) (p=0.468)", variant: "original" },
    { label: "ADVERSARIAL", subtitle: "6 (horse) (p=1.000)", variant: "adversarial" },
    { label: "DEFENDED", subtitle: "5 (dog) (p=0.174)", variant: "defended" },
    { label: "DIFF", subtitle: "|def - adv|", variant: "diff" },
  ];

  const modeLabel = defenseMode === "restore_clean" ? "restore" : "max prob";
  const stackLabel = defenseStack;
  const orderLabel = defenseOrder === "jpeg_noise" ? "JPEG→Noise" : "Noise→JPEG";

  const pageLabel = page === "attacks" ? "атаки" : "защиты";

  return (
    <div className="relative min-h-screen bg-[#060913] text-slate-100">
      <div className="pointer-events-none absolute left-1/2 top-0 h-72 w-72 -translate-x-1/2 rounded-full bg-indigo-600/30 blur-[140px]" />
      <div className="pointer-events-none absolute bottom-0 right-0 h-72 w-72 translate-x-1/4 translate-y-1/4 rounded-full bg-sky-500/20 blur-[160px]" />

      <div className="relative mx-auto flex max-w-[1600px] flex-col gap-6 px-6 py-6">
        <header className="flex flex-wrap items-center justify-between gap-6 rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4 shadow-[0_0_30px_rgba(15,23,42,0.35)]">
          <div>
            <div className="text-sm text-slate-400">NA_project / UI макет — атаки/защиты</div>
            <div className="mt-1 text-base font-semibold">Страница: {pageLabel}</div>
          </div>
          <div className="flex items-center gap-4">
            {(["attacks", "defenses"] as PageKey[]).map((key) => (
              <button
                key={key}
                type="button"
                className={`relative px-2 text-sm font-medium uppercase tracking-wide transition ${
                  page === key ? "text-white" : "text-slate-400 hover:text-slate-200"
                }`}
                onClick={() => setPage(key)}
              >
                {key === "attacks" ? "Атаки" : "Защиты"}
                {page === key && (
                  <span className="absolute -bottom-2 left-0 h-0.5 w-full rounded-full bg-indigo-500" />
                )}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2 text-slate-200">
              <input type="checkbox" checked={grid} onChange={(e) => setGrid(e.target.checked)} />
              Pixel grid
            </label>
            <select
              className="rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1 text-slate-100"
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
            >
              <option value={1}>×1</option>
              <option value={2}>×2</option>
              <option value={3}>×3</option>
              <option value={4}>×4</option>
              <option value={6}>×6</option>
            </select>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[340px_1fr]">
          <aside className="space-y-6">
            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
              <h2 className="text-base font-semibold">Dataset</h2>
              <p className="text-sm text-slate-400">Размер: {datasetSize}</p>
              {metaError && <p className="mt-2 text-xs text-amber-400">{metaError}</p>}
              <div className="mt-4 flex items-center gap-2">
                <Button variant="outline" onClick={() => setIndex((prev) => clampIndex(prev - 1))}>
                  Prev
                </Button>
                <input
                  type="number"
                  min={1}
                  max={datasetSize}
                  className="w-full rounded-lg border border-white/10 bg-white/[0.04] p-2 text-slate-100"
                  value={index}
                  onChange={(e) =>
                    setIndex(clampIndex(safeNumber(Number(e.target.value), 1)))
                  }
                />
                <Button variant="outline" onClick={() => setIndex((prev) => clampIndex(prev + 1))}>
                  Next
                </Button>
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
              <h2 className="text-base font-semibold">Inference</h2>
              <Button className="mt-3" variant="primary" onClick={handleInfer}>
                Run inference
              </Button>
              <div className="mt-4 space-y-2 text-sm text-slate-200">
                <div>Latency: {inferResult?.latency_ms.toFixed(1) ?? "176.9"} ms</div>
                <div>True label: {inferResult?.label ?? 5}</div>
                <ul className="space-y-1 text-slate-300">
                  {(inferResult?.top_k ?? MOCK_INFER.top_k).map((item) => (
                    <li key={item.index}>
                      {item.class} ({item.score.toFixed(3)})
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            {page === "attacks" ? (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <h2 className="text-base font-semibold">Attacks</h2>
                <div className="mt-3 space-y-2 text-sm">
                  {attacks.length === 0 ? (
                    <div className="text-xs text-slate-500">No attacks loaded yet.</div>
                  ) : (
                    attacks.map((name) => (
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
                    ))
                  )}
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
            ) : (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <h2 className="text-base font-semibold">Defenses</h2>
                <div className="mt-3 space-y-3 text-sm">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Режим подбора</span>
                    <select
                      className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-100"
                      value={defenseMode}
                      onChange={(e) => setDefenseMode(e.target.value)}
                    >
                      <option value="restore_clean">Вернуть класс как на чистом</option>
                      <option value="maximize_clean_prob">Макс. вероятность класса на чистом</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Набор защит</span>
                    <select
                      className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-100"
                      value={defenseStack}
                      onChange={(e) => setDefenseStack(e.target.value)}
                    >
                      <option value="jpeg">JPEG</option>
                      <option value="noise">Шум Гаусса</option>
                      <option value="jpeg+noise">JPEG + шум</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Порядок</span>
                    <select
                      className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-100"
                      value={defenseOrder}
                      onChange={(e) => setDefenseOrder(e.target.value)}
                    >
                      <option value="jpeg_noise">JPEG→Noise</option>
                      <option value="noise_jpeg">Noise→JPEG</option>
                    </select>
                  </label>
                  <div className="flex items-end gap-2">
                    <label className="flex flex-1 flex-col gap-1">
                      <span className="text-xs text-slate-400">Seed</span>
                      <input
                        type="number"
                        className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-100"
                        value={defenseSeed}
                        onChange={(e) =>
                          setDefenseSeed(safeNumber(Number(e.target.value), 123))
                        }
                      />
                    </label>
                    <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-[11px] text-emerald-200">
                      без лимита качества
                    </span>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3 text-xs text-slate-300">
                    Цель: добиться правильного ответа любой ценой (подбор JPEG Q и σ).
                  </div>
                </div>
                <Button className="mt-4 w-full" variant="primary" onClick={handleRunDefenses}>
                  Run defenses
                </Button>
                <div className="mt-4 border-t border-white/10 pt-4 text-sm text-slate-300">
                  <div className="flex items-center justify-between text-xs uppercase text-slate-500">
                    <span>Status</span>
                    <span>Progress</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-sm">
                    <span>{defenseStatus}</span>
                    <span>{defenseProgress}%</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-indigo-500 transition-all"
                      style={{ width: `${defenseProgress}%` }}
                    />
                  </div>
                </div>
              </section>
            )}
          </aside>

          <main className="space-y-6">
            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-base font-semibold">Preview (96×96)</h2>
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">index: {index}</span>
                  <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">model: VGG16</span>
                </div>
              </div>
              <div
                className={`mt-4 grid gap-6 ${
                  page === "defenses" ? "sm:grid-cols-2 xl:grid-cols-4" : "md:grid-cols-3"
                }`}
              >
                {(page === "defenses" ? defensePreviewItems : attackPreviewItems).map((item) => (
                  <div key={item.label}>
                    <div className="text-xs uppercase text-slate-400">{item.label}</div>
                    <div className="mt-1 text-xs text-slate-500">{item.subtitle}</div>
                    <div
                      className={`relative mt-3 flex items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] p-3 shadow-inner ${
                        grid ? "pixel-grid" : ""
                      }`}
                    >
                      <span className="absolute right-2 top-2 rounded-full border border-white/10 bg-white/[0.08] px-2 py-0.5 text-[10px] uppercase text-slate-300">
                        mock
                      </span>
                      {item.url ? (
                        <img
                          src={item.url}
                          alt={item.label}
                          style={scaleStyle}
                          className="pixelated rounded-md"
                        />
                      ) : (
                        <div
                          style={scaleStyle}
                          className={`flex items-center justify-center rounded-md text-[10px] uppercase text-slate-300 ${
                            PREVIEW_VARIANTS[item.variant]
                          }`}
                        >
                          96×96 preview
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {page === "attacks" ? (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <h2 className="text-base font-semibold">Attack results</h2>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full text-sm text-slate-200">
                    <thead>
                      <tr className="bg-white/5 text-left text-xs uppercase text-slate-400">
                        <th className="px-3 py-2">Attack</th>
                        <th className="px-3 py-2">Success</th>
                        <th className="px-3 py-2">Pred (before)</th>
                        <th className="px-3 py-2">Pred (after)</th>
                        <th className="px-3 py-2">Linf</th>
                        <th className="px-3 py-2">L2</th>
                        <th className="px-3 py-2">Time</th>
                        <th className="px-3 py-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(job?.results || []).map((row) => (
                        <tr
                          key={row.attack}
                          className={`border-t border-white/10 hover:bg-white/[0.04] ${
                            selectedAttack === row.attack ? "bg-white/[0.06]" : ""
                          }`}
                          onClick={() => setSelectedAttack(row.attack)}
                        >
                          <td className="px-3 py-2 font-medium">{row.attack}</td>
                          <td className="px-3 py-2">
                            {row.success ? "Да" : row.success === false ? "Нет" : "-"}
                          </td>
                          <td className="px-3 py-2">{row.pred_before ?? "-"}</td>
                          <td className="px-3 py-2">{row.pred_after ?? "-"}</td>
                          <td className="px-3 py-2">{row.linf ?? "-"}</td>
                          <td className="px-3 py-2">{row.l2 ?? "-"}</td>
                          <td className="px-3 py-2">{row.time_ms ?? "-"}</td>
                          <td className="px-3 py-2">{row.status}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ) : (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h2 className="text-base font-semibold">Defense results</h2>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">mode: {modeLabel}</span>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">stack: {stackLabel}</span>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">order: {orderLabel}</span>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">seed: {defenseSeed}</span>
                  </div>
                </div>
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full text-sm text-slate-200">
                    <thead>
                      <tr className="bg-white/5 text-left text-xs uppercase text-slate-400">
                        <th className="px-3 py-2">Атака</th>
                        <th className="px-3 py-2 min-w-[140px]">Clean (target)</th>
                        <th className="px-3 py-2 min-w-[140px]">После атаки</th>
                        <th className="px-3 py-2 min-w-[140px]">После защиты</th>
                        <th className="px-3 py-2">JPEG</th>
                        <th className="px-3 py-2">Шум</th>
                        <th className="px-3 py-2">Попытки</th>
                        <th className="px-3 py-2">Деградация</th>
                        <th className="px-3 py-2">Статус</th>
                      </tr>
                    </thead>
                    <tbody>
                      {DEFENSE_ROWS.map((row) => (
                        <tr key={row.attack} className="border-t border-white/10 hover:bg-white/[0.04]">
                          <td className="px-3 py-2 font-medium">{row.attack}</td>
                          <td className="px-3 py-2">{row.clean}</td>
                          <td className="px-3 py-2">{row.attacked}</td>
                          <td className="px-3 py-2">{row.defended}</td>
                          <td className="px-3 py-2">{row.jpeg}</td>
                          <td className="px-3 py-2">{row.noise}</td>
                          <td className="px-3 py-2">{row.attempts}</td>
                          <td className="px-3 py-2">{row.degradation}</td>
                          <td className="px-3 py-2">
                            <span
                              className={`rounded-full px-2 py-1 text-xs ${
                                row.status === "Восстановлено"
                                  ? "border border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
                                  : "border border-rose-400/40 bg-rose-500/10 text-rose-200"
                              }`}
                            >
                              {row.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.04] p-3 text-xs text-slate-300">
                  Примечание: «без лимита качества» означает, что подбор может привести к сильной деградации изображения. Это
                  отображается в столбце «Деградация»
                </div>
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
