import base64
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
import torch

from attacks import ATTACK_ORDER, run_attack
from core.constants import BIN_DIR, CLASSES
from core.dataset import load_train_dataset
from core.model_io import build_model, load_state_dict
from core.settings import load_settings, save_settings
from core.transforms import PIL_AVAILABLE, preprocess, preprocess_x01

if PIL_AVAILABLE:
    from PIL import Image


app = Flask(__name__)

STATE = {
    "images": None,
    "labels": None,
    "model": None,
    "model_path": None,
    "data_dir": None,
}


def _load_state():
    settings = load_settings()
    data_dir = settings.get("data_dir") or str(BIN_DIR)
    model_path = settings.get("model_path")
    STATE["data_dir"] = data_dir
    STATE["model_path"] = model_path

    STATE["images"], STATE["labels"] = load_train_dataset(data_dir)

    if model_path:
        model_path_obj = Path(model_path)
        model = build_model()
        state = load_state_dict(model_path_obj)
        model.load_state_dict(state)
        model.eval()
        STATE["model"] = model


def _get_x01_from_index(idx: int) -> torch.Tensor:
    arr = STATE["images"][idx]
    if PIL_AVAILABLE:
        img = Image.fromarray(arr.transpose(1, 2, 0))
        x01 = preprocess_x01(img)
    else:
        x = torch.from_numpy(arr)
        x01 = preprocess_x01(x)
    return x01.unsqueeze(0)


def _get_tensor_from_index(idx: int) -> torch.Tensor:
    arr = STATE["images"][idx]
    if PIL_AVAILABLE:
        img = Image.fromarray(arr.transpose(1, 2, 0))
        return preprocess(img)
    x = torch.from_numpy(arr).float().div(255.0)
    return x


def _tensor_to_png_bytes(x01: torch.Tensor) -> bytes:
    x = x01.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    img = Image.fromarray(arr).resize((96, 96))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _format_pred(pred: int, p: float) -> str:
    return f"{pred} ({CLASSES[pred]}) (p={p:.3f})"


@app.route("/")
def index():
    if STATE["images"] is None:
        _load_state()
    count = len(STATE["images"])
    model_path = STATE["model_path"] or ""
    data_dir = STATE["data_dir"] or ""
    return render_template("index.html", count=count, model_path=model_path, data_dir=data_dir)


@app.route("/api/indices")
def indices():
    if STATE["images"] is None:
        _load_state()
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 100))
    total = len(STATE["images"])
    indices = list(range(offset, min(offset + limit, total)))
    return jsonify({"indices": indices, "total": total})


@app.route("/api/preview/<int:idx>")
def preview(idx: int):
    if STATE["images"] is None:
        _load_state()
    if not PIL_AVAILABLE:
        return jsonify({"error": "Preview requires Pillow"}), 400
    arr = STATE["images"][idx]
    img = Image.fromarray(arr.transpose(1, 2, 0)).resize((96, 96))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/api/run_inference", methods=["POST"])
def run_inference():
    payload = request.get_json(force=True)
    indices = payload.get("indices", [])
    if not indices:
        return jsonify({"results": []})
    model = STATE.get("model")
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400
    batch = [_get_tensor_from_index(idx) for idx in indices]
    x = torch.stack(batch, dim=0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).tolist()
    results = []
    for offset, (idx, pred) in enumerate(zip(indices, preds)):
        p = float(probs[offset, pred].item())
        results.append({"idx": idx, "pred": _format_pred(pred, p)})
    return jsonify({"results": results})


@app.route("/api/run_attacks", methods=["POST"])
def run_attacks():
    payload = request.get_json(force=True)
    idx = int(payload.get("idx", -1))
    if idx < 0:
        return jsonify({"error": "Missing index"}), 400

    model = STATE.get("model")
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400

    x0 = _get_x01_from_index(idx)
    y_true = int(STATE["labels"][idx])
    params = payload.get("params")
    if not params:
        params = {
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    images = {}
    diffs = {}
    original = None

    if PIL_AVAILABLE:
        original = base64.b64encode(_tensor_to_png_bytes(x0)).decode("utf-8")

    for name in ATTACK_ORDER:
        res = run_attack(
            name=name,
            model=model,
            x0=x0,
            y_true=y_true,
            params=params,
            device=device,
            stop_flag=None,
        )
        results[name] = {
            "pred_before": res.get("pred_before"),
            "p_before": res.get("p_before"),
            "pred_after": res.get("pred_after"),
            "p_after": res.get("p_after"),
            "success": res.get("success"),
            "linf": res.get("linf"),
            "l2": res.get("l2"),
            "time_ms": res.get("time_ms"),
            "error": res.get("error"),
            "stopped": res.get("stopped"),
        }
        x_adv = res.get("x_adv")
        if x_adv is not None and PIL_AVAILABLE:
            img_bytes = _tensor_to_png_bytes(x_adv)
            images[name] = base64.b64encode(img_bytes).decode("utf-8")
            diff_vis = (10.0 * (x_adv - x0).abs()).clamp(0.0, 1.0)
            diff_bytes = _tensor_to_png_bytes(diff_vis)
            diffs[name] = base64.b64encode(diff_bytes).decode("utf-8")

    return jsonify({"results": results, "images": images, "diffs": diffs, "original": original})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    payload = request.get_json(force=True)
    data_dir = payload.get("data_dir")
    model_path = payload.get("model_path")
    settings = load_settings()
    if data_dir:
        settings["data_dir"] = data_dir
    if model_path:
        settings["model_path"] = model_path
    save_settings(settings)
    STATE["images"] = None
    STATE["labels"] = None
    STATE["model"] = None
    STATE["model_path"] = None
    STATE["data_dir"] = None
    return jsonify({"ok": True})


def run():
    _load_state()
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    run()
