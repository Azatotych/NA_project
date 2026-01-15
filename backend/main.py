import asyncio
import json
import threading
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch

from na_core.attacks import ATTACK_ORDER, run_attack
from na_core.constants import BIN_DIR, CLASSES
from na_core.dataset import load_train_dataset
from na_core.model_io import build_model, load_state_dict
from na_core.settings import load_settings, save_settings
from na_core.transforms import PIL_AVAILABLE, preprocess, preprocess_x01

if PIL_AVAILABLE:
    from PIL import Image


RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NA Project API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self):
        self.images = None
        self.labels = None
        self.model = None
        self.model_path = None
        self.data_dir = None

    def load(self):
        settings = load_settings()
        self.data_dir = settings.get("data_dir") or str(BIN_DIR)
        self.model_path = settings.get("model_path")
        self.images, self.labels = load_train_dataset(self.data_dir)
        if self.model_path:
            model_path = Path(self.model_path)
            model = build_model()
            state = load_state_dict(model_path)
            model.load_state_dict(state)
            model.eval()
            self.model = model

    def reset(self):
        self.images = None
        self.labels = None
        self.model = None
        self.model_path = None
        self.data_dir = None


STATE = AppState()
JOBS: Dict[str, Dict[str, Any]] = {}
MAX_ACTIVE_JOBS = 2
MAX_STORED_JOBS = 10


def _ensure_state():
    if STATE.images is None:
        STATE.load()


def _job_dir(job_id: str) -> Path:
    path = RUNS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join([str(row.get(h, "")) for h in headers]))
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_pred(pred: Optional[int], p: Optional[float]) -> Optional[str]:
    if pred is None:
        return None
    if p is None:
        return f"{pred}"
    return f"{pred} ({CLASSES[pred]}) (p={p:.3f})"


def _tensor_to_png(x01: torch.Tensor, size: int = 96) -> bytes:
    x = x01.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    img = Image.fromarray(arr).resize((size, size))
    buffer = Path(tempfile.gettempdir()) / f"na_preview_{time.time_ns()}.png"
    img.save(buffer, format="PNG")
    data = buffer.read_bytes()
    buffer.unlink(missing_ok=True)
    return data


def _default_params() -> Dict[str, Any]:
    return {
        "epsilon": 0.0313725,
        "success_criteria": "true",
        "time_limit": 0.0,
        "fgsm": {"epsilon": 0.0313725},
        "bim": {"epsilon": 0.0313725, "alpha": 0.0078431, "iters": 10},
        "pgd": {"epsilon": 0.0313725, "alpha": 0.0078431, "iters": 20, "random_start": True},
        "deepfool": {"max_iters": 50, "overshoot": 0.02},
        "cw": {"steps": 200, "lr": 0.01, "c": 1.0, "kappa": 0, "binary_search_steps": 5},
        "autoattack": {"epsilon": 0.0313725, "version": "standard"},
    }


def _enqueue_job(job_id: str, index: int, attacks: List[Dict[str, Any]], defense: Optional[str]):
    _ensure_state()
    if STATE.model is None:
        raise RuntimeError("Model not loaded")

    job = JOBS[job_id]
    job["status"] = "running"
    job["progress"] = 0
    job["started_at"] = time.time()

    x0 = _get_x01_from_index(index)
    y_true = int(STATE.labels[index])

    job["x0"] = x0
    job["x_adv"] = {}

    total = len(attacks)
    results = []
    for i, attack in enumerate(attacks, start=1):
        if job.get("stop"):
            job["status"] = "stopped"
            break
        name = attack["name"]
        params = attack.get("params") or _default_params()
        res = run_attack(
            name=name,
            model=STATE.model,
            x0=x0,
            y_true=y_true,
            params=params,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            stop_flag=None,
        )
        job["x_adv"][name] = res.get("x_adv")
        result_row = {
            "attack": name,
            "success": res.get("success"),
            "pred_before": _format_pred(res.get("pred_before"), res.get("p_before")),
            "pred_after": _format_pred(res.get("pred_after"), res.get("p_after")),
            "linf": res.get("linf"),
            "l2": res.get("l2"),
            "time_ms": res.get("time_ms"),
            "status": "ОК" if not res.get("error") else "Ошибка",
        }
        results.append(result_row)
        job["results"] = results
        job["progress"] = int((i / total) * 100)
        _persist_job(job_id)

    job["finished_at"] = time.time()
    if job["status"] != "stopped":
        job["status"] = "done"
    _persist_job(job_id)


def _persist_job(job_id: str):
    job = JOBS[job_id]
    job_dir = _job_dir(job_id)
    results = job.get("results", [])
    _save_json(job_dir / "results.json", results)
    _save_csv(job_dir / "results.csv", results)
    if PIL_AVAILABLE and job.get("x0") is not None:
        original_path = job_dir / "original.png"
        if not original_path.exists():
            original_path.write_bytes(_tensor_to_png(job["x0"]))
        for name, x_adv in job.get("x_adv", {}).items():
            if x_adv is None:
                continue
            attack_dir = job_dir / name
            attack_dir.mkdir(exist_ok=True)
            adv_path = attack_dir / "adv.png"
            diff_path = attack_dir / "diff.png"
            if not adv_path.exists():
                adv_path.write_bytes(_tensor_to_png(x_adv))
            if not diff_path.exists():
                diff_vis = (10.0 * (x_adv - job["x0"]).abs()).clamp(0.0, 1.0)
                diff_path.write_bytes(_tensor_to_png(diff_vis))


def _cleanup_jobs():
    if len(JOBS) <= MAX_STORED_JOBS:
        return
    ordered = sorted(JOBS.items(), key=lambda item: item[1].get("started_at", 0))
    for job_id, _ in ordered[:-MAX_STORED_JOBS]:
        JOBS.pop(job_id, None)
        job_dir = RUNS_DIR / job_id
        if job_dir.exists():
            for path in job_dir.rglob("*"):
                path.unlink(missing_ok=True)
            job_dir.rmdir()


def _reset_jobs():
    for job_id, job in list(JOBS.items()):
        job["stop"] = True
        job["status"] = "stopped"
        job["progress"] = 0
        job["results"] = []
        job_dir = RUNS_DIR / job_id
        if job_dir.exists():
            for path in job_dir.rglob("*"):
                path.unlink(missing_ok=True)
            job_dir.rmdir()


def _get_x01_from_index(idx: int) -> torch.Tensor:
    arr = STATE.images[idx]
    if PIL_AVAILABLE:
        img = Image.fromarray(arr.transpose(1, 2, 0))
        x01 = preprocess_x01(img)
    else:
        x = torch.from_numpy(arr)
        x01 = preprocess_x01(x)
    return x01.unsqueeze(0)


def _get_tensor_from_index(idx: int) -> torch.Tensor:
    arr = STATE.images[idx]
    if PIL_AVAILABLE:
        img = Image.fromarray(arr.transpose(1, 2, 0))
        return preprocess(img)
    x = torch.from_numpy(arr).float().div(255.0)
    return x


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/attacks")
def attacks_list():
    return {
        "attacks": ATTACK_ORDER,
        "presets": {
            "fast": {"epsilon": 0.0156863},
            "default": {"epsilon": 0.0313725},
            "strong": {"epsilon": 0.0627451},
        },
    }


@app.get("/api/v1/defenses")
def defenses_list():
    return {"defenses": []}


@app.get("/api/v1/dataset/info")
def dataset_info():
    _ensure_state()
    return {"size": len(STATE.images), "classes": CLASSES}


@app.get("/api/v1/images/{index}")
def image(index: int, format: str = Query("png")):
    _ensure_state()
    if index < 0 or index >= len(STATE.images):
        raise HTTPException(status_code=404, detail="Index out of range")
    arr = STATE.images[index]
    if format == "raw":
        return JSONResponse({"data": arr.tolist()})
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=400, detail="Preview requires Pillow")
    img = Image.fromarray(arr.transpose(1, 2, 0)).resize((96, 96))
    path = Path(tempfile.gettempdir()) / f"na_image_{time.time_ns()}.png"
    img.save(path, format="PNG")
    return FileResponse(path, media_type="image/png")


@app.post("/api/v1/infer")
def infer(payload: Dict[str, Any]):
    _ensure_state()
    index = int(payload.get("index", -1))
    top_k = int(payload.get("top_k", 5))
    if index < 0 or index >= len(STATE.images):
        raise HTTPException(status_code=404, detail="Index out of range")
    if STATE.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    start = time.time()
    x = _get_tensor_from_index(index).unsqueeze(0)
    with torch.no_grad():
        logits = STATE.model(x)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=top_k, dim=1)
    latency = (time.time() - start) * 1000
    results = []
    for idx, score in zip(indices[0].tolist(), values[0].tolist()):
        results.append({"class": CLASSES[idx], "index": idx, "score": score})
    label = int(STATE.labels[index]) if STATE.labels is not None else None
    return {
        "index": index,
        "label": label,
        "latency_ms": latency,
        "top_k": results,
    }


@app.post("/api/v1/jobs/attack")
def create_attack_job(payload: Dict[str, Any]):
    _ensure_state()
    active_jobs = [job for job in JOBS.values() if job.get("status") == "running"]
    if len(active_jobs) >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=429, detail="Too many active jobs")
    index = int(payload.get("index", -1))
    if index < 0 or index >= len(STATE.images):
        raise HTTPException(status_code=404, detail="Index out of range")
    attacks = payload.get("attacks") or [{"name": name, "params": _default_params()} for name in ATTACK_ORDER]
    defense = payload.get("defense")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "results": [],
        "index": index,
        "created_at": time.time(),
        "defense": defense,
    }
    thread = threading.Thread(target=_enqueue_job, args=(job_id, index, attacks, defense), daemon=True)
    thread.start()
    _cleanup_jobs()
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "results": job.get("results", []),
        "index": job.get("index"),
    }


@app.websocket("/api/v1/ws/jobs/{job_id}")
async def job_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job = JOBS.get(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                await websocket.close()
                break
            await websocket.send_json({
                "status": job.get("status"),
                "progress": job.get("progress"),
                "results": job.get("results", []),
            })
            await websocket.receive_text()
            await asyncio.sleep(0.5)
    except Exception:
        await websocket.close()


@app.get("/api/v1/jobs/{job_id}/artifacts/{attack_name}")
def job_artifacts(job_id: str, attack_name: str, type: str = Query("adv"), format: str = Query("png"), amplify: float = Query(10.0)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=400, detail="Preview requires Pillow")
    job_dir = RUNS_DIR / job_id
    if type == "original":
        path = job_dir / "original.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Original not available")
        return FileResponse(path, media_type="image/png")
    if type == "adv":
        path = job_dir / attack_name / "adv.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not available")
        return FileResponse(path, media_type="image/png")
    if type == "diff":
        if job.get("x0") is None:
            raise HTTPException(status_code=404, detail="Original not available")
        x_adv = job.get("x_adv", {}).get(attack_name)
        if x_adv is None:
            raise HTTPException(status_code=404, detail="Artifact not available")
        diff_vis = (amplify * (x_adv - job["x0"]).abs()).clamp(0.0, 1.0)
        data = _tensor_to_png(diff_vis)
        tmp_path = RUNS_DIR / job_id / f"diff_{attack_name}.png"
        tmp_path.write_bytes(data)
        return FileResponse(tmp_path, media_type="image/png")
    raise HTTPException(status_code=400, detail="Unknown artifact type")


@app.get("/api/v1/jobs/{job_id}/export.csv")
def export_csv(job_id: str):
    path = RUNS_DIR / job_id / "results.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not available")
    return FileResponse(path, media_type="text/csv")


@app.get("/api/v1/jobs/{job_id}/export.json")
def export_json(job_id: str):
    path = RUNS_DIR / job_id / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not available")
    return FileResponse(path, media_type="application/json")


@app.post("/api/v1/jobs/reset")
def reset_jobs():
    _reset_jobs()
    return {"ok": True}


@app.post("/api/v1/settings")
def update_settings(payload: Dict[str, Any]):
    data_dir = payload.get("data_dir")
    model_path = payload.get("model_path")
    settings = load_settings()
    if data_dir:
        settings["data_dir"] = data_dir
    if model_path:
        settings["model_path"] = model_path
    save_settings(settings)
    STATE.reset()
    return {"ok": True}
