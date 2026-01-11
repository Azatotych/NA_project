import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchattacks

    TORCHATTACKS_AVAILABLE = True
    TORCHATTACKS_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    TORCHATTACKS_AVAILABLE = False
    TORCHATTACKS_ERROR = str(exc)

try:
    from autoattack import AutoAttack

    AUTOATTACK_AVAILABLE = True
    AUTOATTACK_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    AUTOATTACK_AVAILABLE = False
    AUTOATTACK_ERROR = str(exc)


ATTACK_ORDER = ["FGSM", "BIM", "PGD", "DeepFool", "C&W", "AutoAttack"]


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def _predict(model: nn.Module, x: torch.Tensor):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = int(probs.argmax(dim=1).item())
    p = float(probs[0, pred].item())
    return pred, p


def _linf_l2(x_adv: torch.Tensor, x0: torch.Tensor):
    diff = (x_adv - x0).detach()
    linf = float(diff.abs().max().item())
    l2 = float(torch.norm(diff.view(diff.size(0), -1), p=2, dim=1).item())
    return linf, l2


def _success(pred_after: int, y_true: int, pred_before: int, criterion: str) -> bool:
    if criterion == "pred":
        return pred_after != pred_before
    return pred_after != y_true


def _parse_time_limit(params: Dict[str, Any]) -> float:
    limit = params.get("time_limit", 0.0)
    try:
        limit = float(limit)
    except Exception:
        limit = 0.0
    return max(limit, 0.0)


def _autoattack_simple(
    norm_model: nn.Module,
    x0: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float,
    bim_alpha: float,
    bim_iters: int,
    pgd_alpha: float,
    pgd_iters: int,
    stop_flag,
    time_limit: float,
    start_time: float,
    progress_cb=None,
):
    total_steps = max(1, 1 + bim_iters + pgd_iters)
    current_step = 0

    def bump(step=1):
        nonlocal current_step
        current_step += step
        if progress_cb:
            progress_cb("AutoAttack", current_step, total_steps)

    def check_stop():
        if stop_flag is not None and stop_flag.is_set():
            return True
        if time_limit and (time.time() - start_time) > time_limit:
            raise TimeoutError("Превышен лимит времени")
        return False

    candidates = []

    if check_stop():
        return x0

    # FGSM candidate
    x_adv = x0.clone().detach().requires_grad_(True)
    logits = norm_model(x_adv)
    loss = F.cross_entropy(logits, y_ref)
    loss.backward()
    x_adv = torch.clamp(x_adv + eps * x_adv.grad.sign(), 0.0, 1.0).detach()
    candidates.append(x_adv)
    bump()

    # BIM candidate
    x_adv = x0.clone().detach()
    for _ in range(bim_iters):
        if check_stop():
            return x_adv.detach()
        x_adv.requires_grad_(True)
        logits = norm_model(x_adv)
        loss = F.cross_entropy(logits, y_ref)
        loss.backward()
        x_adv = x_adv + bim_alpha * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).detach()
        bump()
    candidates.append(x_adv)

    # PGD candidate (random start)
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(pgd_iters):
        if check_stop():
            return x_adv.detach()
        x_adv.requires_grad_(True)
        logits = norm_model(x_adv)
        loss = F.cross_entropy(logits, y_ref)
        loss.backward()
        x_adv = x_adv + pgd_alpha * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).detach()
        bump()
    candidates.append(x_adv)

    # Choose highest loss candidate
    best = candidates[0]
    best_loss = None
    for cand in candidates:
        with torch.no_grad():
            logits = norm_model(cand)
            loss = F.cross_entropy(logits, y_ref)
            loss_val = float(loss.item())
        if best_loss is None or loss_val > best_loss:
            best_loss = loss_val
            best = cand
    return best.detach()


def run_attack(
    name: str,
    model: nn.Module,
    x0: torch.Tensor,
    y_true: int,
    params: Dict[str, Any],
    device: torch.device,
    stop_flag,
    progress_cb=None,
) -> Dict[str, Any]:
    result = {
        "x_adv": None,
        "pred_before": None,
        "p_before": None,
        "pred_after": None,
        "p_after": None,
        "success": False,
        "linf": None,
        "l2": None,
        "time_ms": None,
        "error": None,
        "stopped": False,
        "note": None,
    }

    if stop_flag is not None and stop_flag.is_set():
        result["stopped"] = True
        return result

    start = time.time()
    time_limit = _parse_time_limit(params)

    model = model.to(device)
    x0 = x0.to(device)
    y_true_t = torch.tensor([y_true], dtype=torch.long, device=device)

    norm_model = NormalizedModel(model, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).to(device)
    norm_model.eval()

    with torch.no_grad():
        pred_before, p_before = _predict(norm_model, x0)

    result["pred_before"] = pred_before
    result["p_before"] = p_before

    criterion = params.get("success_criteria", "true")
    y_ref = y_true_t if criterion == "true" else torch.tensor([pred_before], dtype=torch.long, device=device)

    try:
        if name == "FGSM":
            eps = float(params["fgsm"]["epsilon"])
            x_adv = x0.clone().detach().requires_grad_(True)
            logits = norm_model(x_adv)
            loss = F.cross_entropy(logits, y_ref)
            loss.backward()
            x_adv = x_adv + eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
            if progress_cb:
                progress_cb(name, 1, 1)

        elif name == "BIM":
            eps = float(params["bim"]["epsilon"])
            alpha = float(params["bim"]["alpha"])
            iters = int(params["bim"]["iters"])
            x_adv = x0.clone().detach()
            for i in range(iters):
                if stop_flag is not None and stop_flag.is_set():
                    result["stopped"] = True
                    break
                if time_limit and (time.time() - start) > time_limit:
                    raise TimeoutError("Превышен лимит времени")
                x_adv.requires_grad_(True)
                logits = norm_model(x_adv)
                loss = F.cross_entropy(logits, y_ref)
                loss.backward()
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).detach()
                if progress_cb:
                    progress_cb(name, i + 1, iters)
            if result["stopped"]:
                return result

        elif name == "PGD":
            eps = float(params["pgd"]["epsilon"])
            alpha = float(params["pgd"]["alpha"])
            iters = int(params["pgd"]["iters"])
            random_start = bool(params["pgd"]["random_start"])
            if random_start:
                x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            else:
                x_adv = x0.clone().detach()
            for i in range(iters):
                if stop_flag is not None and stop_flag.is_set():
                    result["stopped"] = True
                    break
                if time_limit and (time.time() - start) > time_limit:
                    raise TimeoutError("Превышен лимит времени")
                x_adv.requires_grad_(True)
                logits = norm_model(x_adv)
                loss = F.cross_entropy(logits, y_ref)
                loss.backward()
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).detach()
                if progress_cb:
                    progress_cb(name, i + 1, iters)
            if result["stopped"]:
                return result

        elif name == "DeepFool":
            if not TORCHATTACKS_AVAILABLE:
                raise RuntimeError(
                    "Для атаки DeepFool требуется пакет torchattacks. Установите зависимость и перезапустите программу."
                )
            max_iters = int(params["deepfool"]["max_iters"])
            overshoot = float(params["deepfool"]["overshoot"])
            attacker = torchattacks.DeepFool(norm_model, steps=max_iters, overshoot=overshoot)
            x_adv = attacker(x0, y_true_t)
            if progress_cb:
                progress_cb(name, 1, 1)

        elif name == "C&W":
            if not TORCHATTACKS_AVAILABLE:
                raise RuntimeError(
                    "Для атаки C&W требуется пакет torchattacks. Установите зависимость и перезапустите программу."
                )
            steps = int(params["cw"]["steps"])
            lr = float(params["cw"]["lr"])
            c = float(params["cw"]["c"])
            kappa = float(params["cw"]["kappa"])
            bss = int(params["cw"]["binary_search_steps"])
            try:
                attacker = torchattacks.CW(
                    norm_model,
                    c=c,
                    kappa=kappa,
                    steps=steps,
                    lr=lr,
                    binary_search_steps=bss,
                )
            except TypeError as exc:
                if "binary_search_steps" in str(exc):
                    attacker = torchattacks.CW(norm_model, c=c, kappa=kappa, steps=steps, lr=lr)
                else:
                    raise
            x_adv = attacker(x0, y_true_t)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            if progress_cb:
                progress_cb(name, 1, 1)

        elif name == "AutoAttack":
            eps = float(params["autoattack"]["epsilon"])
            version = params["autoattack"]["version"]
            if AUTOATTACK_AVAILABLE:
                attacker = AutoAttack(norm_model, norm="Linf", eps=eps, version=version, device=device)
                x_adv = attacker.run_standard_evaluation(x0, y_true_t, bs=1)
                if progress_cb:
                    progress_cb(name, 1, 1)
            else:
                x_adv = _autoattack_simple(
                    norm_model=norm_model,
                    x0=x0,
                    y_ref=y_ref,
                    eps=eps,
                    bim_alpha=float(params["bim"]["alpha"]),
                    bim_iters=int(params["bim"]["iters"]),
                    pgd_alpha=float(params["pgd"]["alpha"]),
                    pgd_iters=int(params["pgd"]["iters"]),
                    stop_flag=stop_flag,
                    time_limit=time_limit,
                    start_time=start,
                    progress_cb=progress_cb,
                )
                result["note"] = "Используется упрощенная реализация AutoAttack."

        else:
            raise ValueError(f"Unknown attack: {name}")

        if result["stopped"]:
            return result

        with torch.no_grad():
            pred_after, p_after = _predict(norm_model, x_adv)

        linf, l2 = _linf_l2(x_adv, x0)

        result.update(
            {
                "x_adv": x_adv.detach().cpu(),
                "pred_after": pred_after,
                "p_after": p_after,
                "success": _success(pred_after, int(y_true), pred_before, criterion),
                "linf": linf,
                "l2": l2,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["time_ms"] = int((time.time() - start) * 1000)

    return result


def run_all_attacks(
    model: nn.Module,
    x0: torch.Tensor,
    y_true: int,
    params: Dict[str, Any],
    device: torch.device,
    stop_flag,
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for name in ATTACK_ORDER:
        if stop_flag is not None and stop_flag.is_set():
            results[name] = {"stopped": True, "error": None}
            continue
        results[name] = run_attack(name, model, x0, y_true, params, device, stop_flag)
        if results[name].get("stopped"):
            for rest in ATTACK_ORDER[ATTACK_ORDER.index(name) + 1 :]:
                results[rest] = {"stopped": True, "error": None}
            break
    return results
