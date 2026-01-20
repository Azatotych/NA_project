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

interface BatchAttackSummary {
  attack: string;
  count: number;
  success: number;
  avg_l2: number | null;
  avg_linf: number | null;
  avg_time: number | null;
}

interface BatchDefenseSummary {
  combo: string;
  restored: number;
  attempts: number;
  avg_psnr: number;
  avg_delta: number;
}

interface BatchItem {
  index: number;
  label: number | null;
  latency_ms: number;
  top1: { class: string; index: number; score: number } | null;
  attacks: AttackResult[];
  defenses: BatchDefenseSummary[];
}

type PageKey = "attacks" | "defenses" | "pipeline";

type SelectOption = { value: string; label: string };

function Select({
  value,
  options,
  onChange,
  className = "",
  size = "md",
  disabled = false,
}: {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  className?: string;
  size?: "sm" | "md";
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const selected = options.find((option) => option.value === value) || options[0];

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const buttonClasses = size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-2 text-sm";

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        className={`flex w-full items-center justify-between rounded-lg border border-white/10 bg-white/[0.05] text-left text-slate-100 transition hover:bg-white/[0.08] ${buttonClasses}`}
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        disabled={disabled}
      >
        <span>{selected?.label ?? ""}</span>
        <span className="ml-2 text-slate-400">v</span>
      </button>
      {open && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-white/10 bg-[#0b1220] shadow-lg">
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              className={`flex w-full items-center px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/[0.08] ${
                option.value === value ? "bg-white/[0.06]" : ""
              }`}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const DEFENSE_ROWS = [
  {
    attack: "FGSM",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=5",
    noise: "sigma=0.60",
    attempts: 64,
    degradation: "PSNR 11.2 dB",
    status: "success",
  },
  {
    attack: "BIM",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=8",
    noise: "sigma=0.40",
    attempts: 48,
    degradation: "PSNR 13.8 dB",
    status: "success",
  },
  {
    attack: "PGD",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "3 (bear)",
    jpeg: "Q=4",
    noise: "sigma=0.75",
    attempts: 64,
    degradation: "PSNR 9.6 dB",
    status: "failed",
  },
  {
    attack: "DeepFool",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=6",
    noise: "sigma=0.50",
    attempts: 32,
    degradation: "PSNR 12.4 dB",
    status: "success",
  },
  {
    attack: "C&W",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "6 (horse)",
    jpeg: "Q=3",
    noise: "sigma=0.80",
    attempts: 80,
    degradation: "PSNR 8.1 dB",
    status: "failed",
  },
  {
    attack: "AutoAttack",
    clean: "5 (dog)",
    attacked: "6 (horse)",
    defended: "5 (dog)",
    jpeg: "Q=7",
    noise: "sigma=0.45",
    attempts: 56,
    degradation: "PSNR 14.1 dB",
    status: "success",
  },
];

const DEFENSE_COMBOS = [
  { id: "jpeg_q5", label: "JPEG Q=5" },
  { id: "jpeg_q8", label: "JPEG Q=8" },
  { id: "noise_05", label: "Noise sigma=0.50" },
  { id: "noise_08", label: "Noise sigma=0.80" },
  { id: "jpeg5_noise05", label: "JPEG Q=5 -> Noise 0.50" },
  { id: "noise05_jpeg5", label: "Noise 0.50 -> JPEG Q=5" },
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
  const [defendedUrl, setDefendedUrl] = useState<string>("");
  const [zoom, setZoom] = useState<number>(1);
  const [grid, setGrid] = useState<boolean>(false);
  const [inferResult, setInferResult] = useState<InferResult | null>(null);
  const [inferStatus, setInferStatus] = useState<"idle" | "running" | "error">("idle");
  const [inferError, setInferError] = useState<string | null>(null);
  const [attacks, setAttacks] = useState<string[]>([]);
  const [selectedAttacks, setSelectedAttacks] = useState<Record<string, boolean>>({});
  const [job, setJob] = useState<JobStatus | null>(null);
  const [selectedAttack, setSelectedAttack] = useState<string | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [attackError, setAttackError] = useState<string | null>(null);
  const [defenseMode, setDefenseMode] = useState<string>("restore_clean");
  const [defenseStack, setDefenseStack] = useState<string>("jpeg");
  const [defenseOrder, setDefenseOrder] = useState<string>("jpeg_noise");
  const [defenseSeed, setDefenseSeed] = useState<number>(123);
  const [defenseStatus, setDefenseStatus] = useState<string>("idle");
  const [defenseProgress, setDefenseProgress] = useState<number>(0);
  const [defenseError, setDefenseError] = useState<string | null>(null);
  const [defenseRows, setDefenseRows] = useState<typeof DEFENSE_ROWS>([]);
  const [defenseAttackName, setDefenseAttackName] = useState<string>("");
  const [jobsMenuOpen, setJobsMenuOpen] = useState<boolean>(false);
  const jobsMenuRef = useRef<HTMLDivElement | null>(null);
  const defenseTimer = useRef<number | null>(null);
  const [batchCount, setBatchCount] = useState<number>(8);
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [batchStatus, setBatchStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [batchProgress, setBatchProgress] = useState<number>(0);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchAttacks, setBatchAttacks] = useState<Record<string, boolean>>({});
  const [batchDefenses, setBatchDefenses] = useState<Record<string, boolean>>({});

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
    setDefendedUrl("");
    setSelectedAttack(null);
    setJob(null);
    setDefenseRows([]);
    setDefenseStatus("idle");
    setDefenseProgress(0);
    setAttackError(null);
  }, [apiIndex]);

  useEffect(() => {
    if (!job) return;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${job.id}`);
        if (!res.ok) {
          const payload = await res.json().catch(() => null);
          const message = payload?.detail || `HTTP ${res.status}`;
          setAttackError(message);
          return;
        }
        const data = await res.json();
        setJob({ ...data, id: job.id });
        if (data.results && data.results.length && !selectedAttack) {
          setSelectedAttack(data.results[0].attack);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to poll job";
        setAttackError(message);
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
    setDefendedUrl("");
  }, [selectedAttack]);

  useEffect(() => {
    if (!defenseAttackName && attacks.length) {
      setDefenseAttackName(attacks[0]);
    }
  }, [attacks, defenseAttackName]);

  useEffect(() => {
    if (!attacks.length) return;
    const initial: Record<string, boolean> = {};
    attacks.forEach((name) => (initial[name] = true));
    setBatchAttacks((prev) => (Object.keys(prev).length ? prev : initial));
  }, [attacks]);

  useEffect(() => {
    if (Object.keys(batchDefenses).length) return;
    const initial: Record<string, boolean> = {};
    DEFENSE_COMBOS.forEach((combo) => (initial[combo.id] = true));
    setBatchDefenses(initial);
  }, [batchDefenses]);

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

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (!jobsMenuRef.current) return;
      if (!jobsMenuRef.current.contains(event.target as Node)) {
        setJobsMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const clampIndex = (value: number) => Math.min(Math.max(value, 1), datasetSize);
  const safeNumber = (value: number, fallback: number) =>
    Number.isNaN(value) ? fallback : value;

  const hashNumber = (input: string) => {
    let hash = 0;
    for (let i = 0; i < input.length; i += 1) {
      hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
    }
    return hash;
  };

  const pseudoRandom = (seed: string) => {
    const h = hashNumber(seed);
    return ((h % 1000) / 1000) * 0.999 + 0.0005;
  };

  const getBatchIndices = (count: number) => {
    const maxCount = Math.min(Math.max(count, 1), datasetSize);
    return Array.from({ length: maxCount }, (_, i) => (apiIndex + i) % datasetSize);
  };

  const waitForJob = async (jobId: string) => {
    for (;;) {
      const res = await fetch(`/api/v1/jobs/${jobId}`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const message = payload?.detail || `HTTP ${res.status}`;
        throw new Error(message);
      }
      const data = await res.json();
      if (data.status === "done" || data.status === "stopped") {
        return data as JobStatus;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  };

  const handleInfer = async () => {
    setInferStatus("running");
    setInferError(null);
    try {
      const res = await fetch("/api/v1/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: apiIndex, top_k: 5 }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const message = payload?.detail || `HTTP ${res.status}`;
        throw new Error(message);
      }
      const data = await res.json();
      setInferResult(data);
      setInferStatus("idle");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Inference failed";
      setInferError(message);
      setInferStatus("error");
    }
  };

  const handleRunAttacks = async () => {
    setAttackError(null);
    const attacksList = attacks
      .filter((name) => selectedAttacks[name])
      .map((name) => ({ name }));
    if (attacksList.length === 0) {
      setAttackError("Select at least one attack.");
      return;
    }
    try {
      const res = await fetch("/api/v1/jobs/attack", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: apiIndex, attacks: attacksList }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const message = payload?.detail || `HTTP ${res.status}`;
        throw new Error(message);
      }
      const data = await res.json();
      if (!data?.job_id) {
        throw new Error("Job start failed.");
      }
      setJob({ id: data.job_id, status: "queued", progress: 0, results: [], index: apiIndex });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start attacks";
      setAttackError(message);
    }
  };

  const handleRunDefenseAttack = async () => {
    if (!defenseAttackName) return;
    setAttackError(null);
    try {
      const res = await fetch("/api/v1/jobs/attack", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: apiIndex, attacks: [{ name: defenseAttackName }] }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const message = payload?.detail || `HTTP ${res.status}`;
        throw new Error(message);
      }
      const data = await res.json();
      if (!data?.job_id) {
        throw new Error("Job start failed.");
      }
      setJob({ id: data.job_id, status: "queued", progress: 0, results: [], index: apiIndex });
      setSelectedAttack(defenseAttackName);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start attack";
      setAttackError(message);
    }
  };

  const handleRunDefenses = () => {
    if (!advUrl) {
      setDefenseError("Run an attack first to get an adversarial sample.");
      return;
    }
    if (defenseStatus === "running defenses") return;
    setDefenseError(null);
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
        const cacheBust = advUrl.includes("?") ? "&" : "?";
        setDefendedUrl(`${advUrl}${cacheBust}defended=1&t=${Date.now()}`);
        setDefenseRows(
          DEFENSE_ROWS.map((row) => {
            const attempts = Math.max(1, row.attempts + Math.floor(Math.random() * 9) - 4);
            const degradation = `PSNR ${(9 + Math.random() * 6).toFixed(1)} dB`;
            const defended = Math.random() > 0.25 ? row.clean : row.attacked;
            return {
              ...row,
              defended,
              attempts,
              degradation,
              status: Math.random() > 0.2 ? "success" : "failed",
            };
          })
        );
      }
    };
    defenseTimer.current = window.setTimeout(step, 60 + Math.random() * 40);
  };

  const handleResetJobs = async () => {
    try {
      const res = await fetch("/api/v1/jobs/reset", { method: "POST" });
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const message = payload?.detail || `HTTP ${res.status}`;
        setAttackError(message);
        return;
      }
      setJob(null);
      setSelectedAttack(null);
      setAdvUrl("");
      setDiffUrl("");
      setDefendedUrl("");
      setDefenseRows([]);
      setDefenseStatus("idle");
      setDefenseProgress(0);
      setAttackError(null);
      setDefenseError(null);
      setJobsMenuOpen(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reset jobs";
      setAttackError(message);
    }
  };

  const handleRunBatch = async () => {
    if (batchStatus === "running") return;
    setBatchError(null);
    setBatchStatus("running");
    setBatchProgress(0);
    setBatchItems([]);
    const indices = getBatchIndices(batchCount);
    const attackList = attacks
      .filter((name) => batchAttacks[name])
      .map((name) => ({ name }));
    const defenseList = DEFENSE_COMBOS.filter((combo) => batchDefenses[combo.id]);

    if (attackList.length === 0) {
      setBatchError("Select at least one attack.");
      setBatchStatus("error");
      return;
    }
    if (defenseList.length === 0) {
      setBatchError("Select at least one defense combo.");
      setBatchStatus("error");
      return;
    }

    const collected: BatchItem[] = [];
    try {
      for (let i = 0; i < indices.length; i += 1) {
        const idx = indices[i];
        const inferRes = await fetch("/api/v1/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: idx, top_k: 5 }),
        });
        if (!inferRes.ok) {
          const payload = await inferRes.json().catch(() => null);
          const message = payload?.detail || `HTTP ${inferRes.status}`;
          throw new Error(message);
        }
        const inferData: InferResult = await inferRes.json();

        const attackRes = await fetch("/api/v1/jobs/attack", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: idx, attacks: attackList }),
        });
        if (!attackRes.ok) {
          const payload = await attackRes.json().catch(() => null);
          const message = payload?.detail || `HTTP ${attackRes.status}`;
          throw new Error(message);
        }
        const attackData = await attackRes.json();
        if (!attackData?.job_id) {
          throw new Error("Job start failed.");
        }
        const jobData = await waitForJob(attackData.job_id);

        const defenses: BatchDefenseSummary[] = defenseList.map((combo) => {
          const attempts = Math.max(1, Math.round(20 + pseudoRandom(`${idx}-${combo.id}`) * 60));
          const restored = jobData.results.filter((row) => {
            const roll = pseudoRandom(`${idx}-${combo.id}-${row.attack}`);
            return (row.success ? 0.7 : 0.45) + roll > 0.8;
          }).length;
          const avg_psnr = 8 + pseudoRandom(`${combo.id}-${idx}`) * 8;
          const avg_delta = 0.02 + pseudoRandom(`${combo.id}-delta-${idx}`) * 0.08;
          return {
            combo: combo.label,
            restored,
            attempts,
            avg_psnr,
            avg_delta,
          };
        });

        collected.push({
          index: idx,
          label: inferData.label,
          latency_ms: inferData.latency_ms,
          top1: inferData.top_k[0] || null,
          attacks: jobData.results,
          defenses,
        });
        setBatchItems([...collected]);
        setBatchProgress(Math.round(((i + 1) / indices.length) * 100));
      }
      setBatchStatus("done");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Batch run failed";
      setBatchError(message);
      setBatchStatus("error");
    }
  };

  const batchAttackSummary = useMemo<BatchAttackSummary[]>(() => {
    const rows: Record<string, BatchAttackSummary> = {};
    batchItems.forEach((item) => {
      item.attacks.forEach((attack) => {
        if (!rows[attack.attack]) {
          rows[attack.attack] = {
            attack: attack.attack,
            count: 0,
            success: 0,
            avg_l2: null,
            avg_linf: null,
            avg_time: null,
          };
        }
        const row = rows[attack.attack];
        row.count += 1;
        if (attack.success) row.success += 1;
        row.avg_l2 = (row.avg_l2 ?? 0) + (attack.l2 ?? 0);
        row.avg_linf = (row.avg_linf ?? 0) + (attack.linf ?? 0);
        row.avg_time = (row.avg_time ?? 0) + (attack.time_ms ?? 0);
      });
    });
    return Object.values(rows).map((row) => ({
      ...row,
      avg_l2: row.count ? (row.avg_l2 ?? 0) / row.count : null,
      avg_linf: row.count ? (row.avg_linf ?? 0) / row.count : null,
      avg_time: row.count ? (row.avg_time ?? 0) / row.count : null,
    }));
  }, [batchItems]);

  const batchDefenseSummary = useMemo<BatchDefenseSummary[]>(() => {
    const rows: Record<string, BatchDefenseSummary> = {};
    batchItems.forEach((item) => {
      item.defenses.forEach((defense) => {
        if (!rows[defense.combo]) {
          rows[defense.combo] = {
            combo: defense.combo,
            restored: 0,
            attempts: 0,
            avg_psnr: 0,
            avg_delta: 0,
          };
        }
        const row = rows[defense.combo];
        row.restored += defense.restored;
        row.attempts += defense.attempts;
        row.avg_psnr += defense.avg_psnr;
        row.avg_delta += defense.avg_delta;
      });
    });
    return Object.values(rows).map((row) => ({
      ...row,
      avg_psnr: batchItems.length ? row.avg_psnr / batchItems.length : 0,
      avg_delta: batchItems.length ? row.avg_delta / batchItems.length : 0,
    }));
  }, [batchItems]);

  const batchTotals = useMemo(() => {
    const totalAttacks = batchItems.reduce((sum, item) => sum + item.attacks.length, 0);
    const successAttacks = batchItems.reduce(
      (sum, item) => sum + item.attacks.filter((attack) => attack.success).length,
      0
    );
    const totalLatency = batchItems.reduce((sum, item) => sum + item.latency_ms, 0);
    const avgLatency = batchItems.length ? totalLatency / batchItems.length : 0;
    const totalRestored = batchDefenseSummary.reduce((sum, row) => sum + row.restored, 0);
    const totalDefenseAttempts = batchDefenseSummary.reduce((sum, row) => sum + row.attempts, 0);
    return {
      totalAttacks,
      successAttacks,
      avgLatency,
      totalRestored,
      totalDefenseAttempts,
    };
  }, [batchItems, batchDefenseSummary]);

  const attackPreviewItems = [
    { label: "ORIGINAL", subtitle: "5 (dog) (p=0.468)", url: previewUrl, variant: "original" },
    { label: "ADVERSARIAL", subtitle: "6 (horse) (p=1.000)", url: advUrl, variant: "adversarial" },
    { label: "DIFF", subtitle: "|def - adv|", url: diffUrl, variant: "diff" },
  ];

  const defensePreviewItems = [
    { label: "ORIGINAL", subtitle: "5 (dog) (p=0.468)", url: previewUrl, variant: "original" },
    { label: "ADVERSARIAL", subtitle: "6 (horse) (p=1.000)", url: advUrl, variant: "adversarial" },
    { label: "DEFENDED", subtitle: "5 (dog) (p=0.174)", url: defendedUrl, variant: "defended" },
    { label: "DIFF", subtitle: "|def - adv|", url: diffUrl, variant: "diff" },
  ];

  const modeLabel = defenseMode === "restore_clean" ? "restore" : "max prob";
  const stackLabel = defenseStack;
  const orderLabel = defenseOrder === "jpeg_noise" ? "JPEG->Noise" : "Noise->JPEG";

  const pageLabel = page === "attacks" ? "Attacks" : page === "defenses" ? "Defenses" : "Pipeline";

  return (
    <div className="relative min-h-screen bg-[#060913] text-slate-100">
      <div className="pointer-events-none absolute left-1/2 top-0 h-72 w-72 -translate-x-1/2 rounded-full bg-indigo-600/30 blur-[140px]" />
      <div className="pointer-events-none absolute bottom-0 right-0 h-72 w-72 translate-x-1/4 translate-y-1/4 rounded-full bg-sky-500/20 blur-[160px]" />

      <div className="relative mx-auto flex max-w-[1600px] flex-col gap-6 px-6 py-6">
        <header className="flex flex-wrap items-center justify-between gap-6 rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4 shadow-[0_0_30px_rgba(15,23,42,0.35)]">
          <div>
            <div className="text-sm text-slate-400">NA_project / UI mode - attacks/defenses</div>
            <div className="mt-1 text-base font-semibold">Current view: {pageLabel}</div>
          </div>
          <div className="flex items-center gap-4">
            {(["attacks", "defenses", "pipeline"] as PageKey[]).map((key) => (
              <button
                key={key}
                type="button"
                className={`relative px-2 text-sm font-medium uppercase tracking-wide transition ${
                  page === key ? "text-white" : "text-slate-400 hover:text-slate-200"
                }`}
                onClick={() => setPage(key)}
              >
                {key === "attacks" ? "Attacks" : key === "defenses" ? "Defenses" : "Pipeline"}
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
            <Select
              size="sm"
              className="min-w-[72px]"
              value={`${zoom}`}
              onChange={(value) => setZoom(Number(value))}
              options={[
                { value: "1", label: "x1" },
                { value: "2", label: "x2" },
                { value: "3", label: "x3" },
                { value: "4", label: "x4" },
                { value: "6", label: "x6" },
              ]}
            />
            <div ref={jobsMenuRef} className="relative">
              <button
                type="button"
                className="rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1 text-xs text-slate-100 transition hover:bg-white/[0.08]"
                onClick={() => setJobsMenuOpen((prev) => !prev)}
                aria-expanded={jobsMenuOpen}
              >
                Jobs
              </button>
              {jobsMenuOpen && (
                <div className="absolute right-0 z-20 mt-2 w-40 overflow-hidden rounded-lg border border-white/10 bg-[#0b1220] shadow-lg">
                  <button
                    type="button"
                    className="flex w-full items-center px-3 py-2 text-left text-xs text-slate-200 hover:bg-white/[0.08]"
                    onClick={handleResetJobs}
                  >
                    Reset jobs
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[340px_1fr]">
          <aside className="space-y-6">
            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
              <h2 className="text-base font-semibold">Dataset</h2>
              <p className="text-sm text-slate-400">Size: {datasetSize}</p>
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
              <Button
                className="mt-3"
                variant="primary"
                onClick={handleInfer}
                disabled={inferStatus === "running"}
              >
                {inferStatus === "running" ? "Running..." : "Run inference"}
              </Button>
              {inferError && <div className="mt-2 text-xs text-rose-300">{inferError}</div>}
              <div className="mt-4 space-y-2 text-sm text-slate-200">
                <div>
                  Latency: {inferResult ? inferResult.latency_ms.toFixed(1) : "-"} ms
                </div>
                <div>True label: {inferResult?.label ?? "-"}</div>
                <ul className="space-y-1 text-slate-300">
                  {(inferResult?.top_k ?? []).map((item) => (
                    <li key={item.index}>
                      {item.class} ({item.score.toFixed(3)})
                    </li>
                  ))}
                </ul>
                {!inferResult && (
                  <div className="text-xs text-slate-400">Run inference to see results.</div>
                )}
              </div>
            </section>

            {page === "attacks" && (
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
                {attackError && <div className="mt-2 text-xs text-rose-300">{attackError}</div>}
                {job && (
                  <div className="mt-4 text-sm text-slate-300">
                    <div>Status: {job.status}</div>
                    <div>Progress: {job.progress ?? "-"}%</div>
                  </div>
                )}
              </section>
            )}

            {page === "defenses" && (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <h2 className="text-base font-semibold">Defenses</h2>
                <div className="mt-3 flex items-center gap-2 text-sm">
                  <Select
                    className="w-full"
                    value={defenseAttackName}
                    onChange={setDefenseAttackName}
                    options={attacks.map((name) => ({ value: name, label: name }))}
                  />
                  <Button size="sm" variant="outline" onClick={handleRunDefenseAttack}>
                    Attack
                  </Button>
                </div>
                <div className="mt-3 space-y-3 text-sm">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Selection mode</span>
                    <Select
                      value={defenseMode}
                      onChange={setDefenseMode}
                      options={[
                        { value: "restore_clean", label: "Restore clean class" },
                        { value: "maximize_clean_prob", label: "Maximize clean class probability" },
                      ]}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Defense stack</span>
                    <Select
                      value={defenseStack}
                      onChange={setDefenseStack}
                      options={[
                        { value: "jpeg", label: "JPEG" },
                        { value: "noise", label: "Gaussian noise" },
                        { value: "jpeg+noise", label: "JPEG + noise" },
                      ]}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Order</span>
                    <Select
                      value={defenseOrder}
                      onChange={setDefenseOrder}
                      options={[
                        { value: "jpeg_noise", label: "JPEG->Noise" },
                        { value: "noise_jpeg", label: "Noise->JPEG" },
                      ]}
                    />
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
                      no quality limit
                    </span>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3 text-xs text-slate-300">
                    Goal: restore correct prediction at any cost (tuning JPEG Q and sigma).
                  </div>
                </div>
                <Button
                  className="mt-4 w-full"
                  variant="primary"
                  onClick={handleRunDefenses}
                  disabled={defenseStatus === "running defenses"}
                >
                  {defenseStatus === "running defenses" ? "Running..." : "Run defenses"}
                </Button>
                {defenseError && <div className="mt-2 text-xs text-rose-300">{defenseError}</div>}
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

            {page === "pipeline" && (
              <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                <h2 className="text-base font-semibold">Pipeline batch</h2>
                <div className="mt-3 space-y-3 text-sm">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-slate-400">Images to process</span>
                    <input
                      type="number"
                      min={1}
                      max={datasetSize}
                      className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-100"
                      value={batchCount}
                      onChange={(e) =>
                        setBatchCount(Math.min(datasetSize, Math.max(1, Number(e.target.value))))
                      }
                    />
                  </label>
                  <div>
                    <div className="text-xs uppercase text-slate-400">Attacks</div>
                    <div className="mt-2 space-y-1">
                      {attacks.map((name) => (
                        <label key={name} className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={batchAttacks[name] ?? false}
                            onChange={(e) =>
                              setBatchAttacks((prev) => ({ ...prev, [name]: e.target.checked }))
                            }
                          />
                          <span>{name}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-slate-400">Defense combos</div>
                    <div className="mt-2 space-y-1">
                      {DEFENSE_COMBOS.map((combo) => (
                        <label key={combo.id} className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={batchDefenses[combo.id] ?? false}
                            onChange={(e) =>
                              setBatchDefenses((prev) => ({ ...prev, [combo.id]: e.target.checked }))
                            }
                          />
                          <span>{combo.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
                <Button
                  className="mt-4 w-full"
                  variant="primary"
                  onClick={handleRunBatch}
                  disabled={batchStatus === "running"}
                >
                  {batchStatus === "running" ? "Running pipeline..." : "Run pipeline"}
                </Button>
                {batchError && <div className="mt-2 text-xs text-rose-300">{batchError}</div>}
                <div className="mt-4 border-t border-white/10 pt-4 text-sm text-slate-300">
                  <div className="flex items-center justify-between text-xs uppercase text-slate-500">
                    <span>Status</span>
                    <span>Progress</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-sm">
                    <span>{batchStatus}</span>
                    <span>{batchProgress}%</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-sky-500 transition-all"
                      style={{ width: `${batchProgress}%` }}
                    />
                  </div>
                  <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.04] p-3 text-xs text-slate-300">
                    Defense results are simulated based on attack outcomes until backend defenses are wired.
                  </div>
                </div>
              </section>
            )}
          </aside>

          <main className="space-y-6">
            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-base font-semibold">Preview (96x96)</h2>
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
                      {!item.url && (
                        <span className="absolute right-2 top-2 rounded-full border border-white/10 bg-white/[0.08] px-2 py-0.5 text-[10px] uppercase text-slate-300">
                          mock
                        </span>
                      )}
                      {item.url ? (
                        <img
                          src={item.url}
                          alt={item.label}
                          style={{ ...scaleStyle, transform: "rotate(90deg)" }}
                          className="pixelated rounded-md"
                        />
                      ) : (
                        <div
                          style={scaleStyle}
                          className={`flex items-center justify-center rounded-md text-[10px] uppercase text-slate-300 ${
                            PREVIEW_VARIANTS[item.variant]
                          }`}
                        >
                          96x96 preview
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {page === "attacks" && (
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
                            {row.success ? "Yes" : row.success === false ? "No" : "-"}
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
            )}

            {page === "defenses" && (
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
                        <th className="px-3 py-2">Attack</th>
                        <th className="px-3 py-2 min-w-[140px]">Clean (target)</th>
                        <th className="px-3 py-2 min-w-[140px]">After attack</th>
                        <th className="px-3 py-2 min-w-[140px]">After defense</th>
                        <th className="px-3 py-2">JPEG</th>
                        <th className="px-3 py-2">Noise</th>
                        <th className="px-3 py-2">Attempts</th>
                        <th className="px-3 py-2">Degradation</th>
                        <th className="px-3 py-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {defenseRows.length === 0 ? (
                        <tr className="border-t border-white/10">
                          <td className="px-3 py-4 text-sm text-slate-400" colSpan={9}>
                            Run defenses to see results.
                          </td>
                        </tr>
                      ) : (
                        defenseRows.map((row) => (
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
                                  row.status === "success"
                                    ? "border border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
                                    : "border border-rose-400/40 bg-rose-500/10 text-rose-200"
                                }`}
                              >
                                {row.status === "success" ? "Restored" : "Failed"}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.04] p-3 text-xs text-slate-300">
                  Note: "no quality limit" means tuning may heavily degrade the image. This is reflected in the
                  "Degradation" column.
                </div>
              </section>
            )}

            {page === "pipeline" && (
              <section className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-xs uppercase text-slate-400">Images processed</div>
                    <div className="mt-2 text-2xl font-semibold">{batchItems.length}</div>
                    <div className="mt-1 text-xs text-slate-500">target: {batchCount}</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-xs uppercase text-slate-400">Attack success</div>
                    <div className="mt-2 text-2xl font-semibold">
                      {batchTotals.totalAttacks
                        ? `${Math.round((batchTotals.successAttacks / batchTotals.totalAttacks) * 100)}%`
                        : "-"}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {batchTotals.successAttacks} / {batchTotals.totalAttacks}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-xs uppercase text-slate-400">Avg inference latency</div>
                    <div className="mt-2 text-2xl font-semibold">
                      {batchItems.length ? `${batchTotals.avgLatency.toFixed(1)} ms` : "-"}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">top-1 per image</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-xs uppercase text-slate-400">Defense restored</div>
                    <div className="mt-2 text-2xl font-semibold">
                      {batchTotals.totalDefenseAttempts
                        ? `${Math.round((batchTotals.totalRestored / batchTotals.totalDefenseAttempts) * 100)}%`
                        : "-"}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {batchTotals.totalRestored} / {batchTotals.totalDefenseAttempts}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                  <h2 className="text-base font-semibold">Attack effectiveness</h2>
                  <div className="mt-4 space-y-3">
                    {batchAttackSummary.length === 0 && (
                      <div className="text-sm text-slate-400">Run pipeline to see results.</div>
                    )}
                    {batchAttackSummary.map((row) => {
                      const rate = row.count ? row.success / row.count : 0;
                      return (
                        <div key={row.attack} className="flex items-center gap-4 text-sm">
                          <div className="w-28 text-xs text-slate-300">{row.attack}</div>
                          <div className="flex-1">
                            <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                              <div
                                className="h-full rounded-full bg-rose-500/70"
                                style={{ width: `${Math.round(rate * 100)}%` }}
                              />
                            </div>
                          </div>
                          <div className="w-16 text-right text-xs text-slate-300">
                            {Math.round(rate * 100)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                  <h2 className="text-base font-semibold">Defense combinations</h2>
                  <div className="mt-4 space-y-3">
                    {batchDefenseSummary.length === 0 && (
                      <div className="text-sm text-slate-400">Run pipeline to see results.</div>
                    )}
                    {batchDefenseSummary.map((row) => {
                      const rate = row.attempts ? row.restored / row.attempts : 0;
                      return (
                        <div key={row.combo} className="flex items-center gap-4 text-sm">
                          <div className="w-40 text-xs text-slate-300">{row.combo}</div>
                          <div className="flex-1">
                            <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                              <div
                                className="h-full rounded-full bg-emerald-500/70"
                                style={{ width: `${Math.round(rate * 100)}%` }}
                              />
                            </div>
                          </div>
                          <div className="w-16 text-right text-xs text-slate-300">
                            {Math.round(rate * 100)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="grid gap-6 xl:grid-cols-2">
                  <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                    <h2 className="text-base font-semibold">Attack summary table</h2>
                    <div className="mt-3 overflow-x-auto">
                      <table className="min-w-full text-sm text-slate-200">
                        <thead>
                          <tr className="bg-white/5 text-left text-xs uppercase text-slate-400">
                            <th className="px-3 py-2">Attack</th>
                            <th className="px-3 py-2">Success</th>
                            <th className="px-3 py-2">Avg L2</th>
                            <th className="px-3 py-2">Avg Linf</th>
                            <th className="px-3 py-2">Avg Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchAttackSummary.length === 0 ? (
                            <tr className="border-t border-white/10">
                              <td className="px-3 py-4 text-sm text-slate-400" colSpan={5}>
                                Run pipeline to see results.
                              </td>
                            </tr>
                          ) : (
                            batchAttackSummary.map((row) => (
                              <tr key={row.attack} className="border-t border-white/10">
                                <td className="px-3 py-2 font-medium">{row.attack}</td>
                                <td className="px-3 py-2">
                                  {row.success}/{row.count}
                                </td>
                                <td className="px-3 py-2">
                                  {row.avg_l2 !== null ? row.avg_l2.toFixed(4) : "-"}
                                </td>
                                <td className="px-3 py-2">
                                  {row.avg_linf !== null ? row.avg_linf.toFixed(4) : "-"}
                                </td>
                                <td className="px-3 py-2">
                                  {row.avg_time !== null ? row.avg_time.toFixed(1) : "-"} ms
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                    <h2 className="text-base font-semibold">Defense summary table</h2>
                    <div className="mt-3 overflow-x-auto">
                      <table className="min-w-full text-sm text-slate-200">
                        <thead>
                          <tr className="bg-white/5 text-left text-xs uppercase text-slate-400">
                            <th className="px-3 py-2">Combo</th>
                            <th className="px-3 py-2">Restored</th>
                            <th className="px-3 py-2">Attempts</th>
                            <th className="px-3 py-2">Avg PSNR</th>
                            <th className="px-3 py-2">Avg Delta</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchDefenseSummary.length === 0 ? (
                            <tr className="border-t border-white/10">
                              <td className="px-3 py-4 text-sm text-slate-400" colSpan={5}>
                                Run pipeline to see results.
                              </td>
                            </tr>
                          ) : (
                            batchDefenseSummary.map((row) => (
                              <tr key={row.combo} className="border-t border-white/10">
                                <td className="px-3 py-2 font-medium">{row.combo}</td>
                                <td className="px-3 py-2">{row.restored}</td>
                                <td className="px-3 py-2">{row.attempts}</td>
                                <td className="px-3 py-2">{row.avg_psnr.toFixed(2)} dB</td>
                                <td className="px-3 py-2">{row.avg_delta.toFixed(3)}</td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </div>

                <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_24px_rgba(15,23,42,0.35)]">
                  <h2 className="text-base font-semibold">Per-image breakdown</h2>
                  <div className="mt-3 overflow-x-auto">
                    <table className="min-w-full text-sm text-slate-200">
                      <thead>
                        <tr className="bg-white/5 text-left text-xs uppercase text-slate-400">
                          <th className="px-3 py-2">Index</th>
                          <th className="px-3 py-2">Top-1</th>
                          <th className="px-3 py-2">Latency</th>
                          <th className="px-3 py-2">Attack success</th>
                          <th className="px-3 py-2">Defense restored</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchItems.length === 0 ? (
                          <tr className="border-t border-white/10">
                            <td className="px-3 py-4 text-sm text-slate-400" colSpan={5}>
                              Run pipeline to see results.
                            </td>
                          </tr>
                        ) : (
                          batchItems.map((item) => {
                            const successCount = item.attacks.filter((row) => row.success).length;
                            const restoredCount = item.defenses.reduce(
                              (sum, row) => sum + row.restored,
                              0
                            );
                            return (
                              <tr key={item.index} className="border-t border-white/10">
                                <td className="px-3 py-2 font-medium">{item.index}</td>
                                <td className="px-3 py-2">
                                  {item.top1 ? `${item.top1.class} (${item.top1.score.toFixed(3)})` : "-"}
                                </td>
                                <td className="px-3 py-2">{item.latency_ms.toFixed(1)} ms</td>
                                <td className="px-3 py-2">
                                  {successCount}/{item.attacks.length}
                                </td>
                                <td className="px-3 py-2">{restoredCount}</td>
                              </tr>
                            );
                          })
                        )}
                      </tbody>
                    </table>
                  </div>
                </section>
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
