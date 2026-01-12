import io
import threading
from typing import Optional, TYPE_CHECKING

import torch

from attacks import ATTACK_ORDER, NormalizedModel, run_attack
from core.constants import CLASSES
from core.transforms import PIL_AVAILABLE

try:
    import PySimpleGUI as sg

    PSG_AVAILABLE = True
except Exception:
    PSG_AVAILABLE = False

if PIL_AVAILABLE:
    from PIL import Image

if TYPE_CHECKING:
    from ui.main_window import App


class AttacksWindow:
    def __init__(self, app: "App"):
        self.app = app
        self.current_idx = None
        self.current_x0 = None
        self.current_y = None
        self.results = {}
        self.running = False
        self.stop_flag = threading.Event()
        self._blink_on = False
        self._blink_show_adv = False

        self.diff_mode = "signed"
        self.gain = 10.0
        self.selected_attack = ATTACK_ORDER[0] if ATTACK_ORDER else None

        if not PSG_AVAILABLE:
            from tkinter import messagebox

            messagebox.showerror("Ошибка", "PySimpleGUI не установлен.")
            return

        self._thread = threading.Thread(target=self._run_ui, daemon=True)
        self._thread.start()

    def _run_ui(self):
        sg.theme("DarkGrey13")
        header_col = [
            [sg.Text("Индекс:"), sg.Text("-", key="IDX")],
            [sg.Text("Истинная метка:"), sg.Text("-", key="LABEL")],
            [sg.Text("Предсказание (до):"), sg.Text("-", key="PRED")],
            [sg.Button("Синхронизировать", key="SYNC")],
        ]

        params_col = [
            [sg.Text("Устройство"), sg.Combo(["Авто", "CPU", "CUDA"], default_value="Авто", key="DEVICE")],
            [sg.Text("Эпсилон (общий)")],
            [sg.Input("0.0313725", key="COMMON_EPS")],
            [sg.Text("Лимит времени (сек)")],
            [sg.Input("", key="TIME_LIMIT")],
            [sg.Text("Критерий успеха")],
            [sg.Radio("Истинная метка", "CRIT", default=True, key="CRIT_TRUE")],
            [sg.Radio("Предсказание до", "CRIT", key="CRIT_PRED")],
            [sg.Frame("FGSM", [[sg.Text("epsilon"), sg.Input("", key="FGSM_EPS")]])],
            [
                sg.Frame(
                    "BIM",
                    [
                        [sg.Text("epsilon"), sg.Input("", key="BIM_EPS")],
                        [sg.Text("alpha"), sg.Input("0.0078431", key="BIM_ALPHA")],
                        [sg.Text("iters"), sg.Input("10", key="BIM_ITERS")],
                    ],
                )
            ],
            [
                sg.Frame(
                    "PGD",
                    [
                        [sg.Text("epsilon"), sg.Input("", key="PGD_EPS")],
                        [sg.Text("alpha"), sg.Input("0.0078431", key="PGD_ALPHA")],
                        [sg.Text("iters"), sg.Input("20", key="PGD_ITERS")],
                        [sg.Checkbox("Случайный старт", default=True, key="PGD_RANDOM")],
                    ],
                )
            ],
            [
                sg.Frame(
                    "DeepFool",
                    [
                        [sg.Text("max_iters"), sg.Input("50", key="DF_ITERS")],
                        [sg.Text("overshoot"), sg.Input("0.02", key="DF_OVERSHOOT")],
                    ],
                )
            ],
            [
                sg.Frame(
                    "C&W",
                    [
                        [sg.Text("steps"), sg.Input("200", key="CW_STEPS")],
                        [sg.Text("lr"), sg.Input("0.01", key="CW_LR")],
                        [sg.Text("c"), sg.Input("1.0", key="CW_C")],
                        [sg.Text("kappa"), sg.Input("0", key="CW_KAPPA")],
                        [sg.Text("binary_search_steps"), sg.Input("5", key="CW_BSS")],
                    ],
                )
            ],
            [
                sg.Frame(
                    "AutoAttack",
                    [
                        [sg.Text("epsilon"), sg.Input("", key="AA_EPS")],
                        [sg.Text("version"), sg.Combo(["standard", "plus"], default_value="standard", key="AA_VERSION")],
                    ],
                )
            ],
        ]

        run_col = [
            [sg.Button("Запуск", key="RUN"), sg.Button("Остановить", key="STOP", disabled=True)],
            [sg.Text("Выполнение: 0/6", key="PROGRESS")],
            [sg.Text("Прогресс атаки: 0%", key="ATTACK_PROGRESS")],
            [sg.ProgressBar(100, orientation="h", size=(20, 10), key="PROGRESS_BAR")],
        ]

        log_col = [[sg.Multiline("", size=(30, 8), key="LOG", disabled=True)]]

        results_table = sg.Table(
            values=[[name, "-", "-", "-", "-", "-", "-", "-"] for name in ATTACK_ORDER],
            headings=["Атака", "До", "После", "Успех", "Linf", "L2", "Время", "Статус"],
            key="RESULTS",
            enable_events=True,
            auto_size_columns=False,
            col_widths=[10, 16, 16, 6, 6, 6, 8, 10],
            num_rows=6,
        )

        preview_controls = [
            [
                sg.Text("Difference:"),
                sg.Radio("Signed", "DIFF", default=True, key="DIFF_SIGNED"),
                sg.Radio("Abs", "DIFF", key="DIFF_ABS"),
                sg.Text("Gain:"),
                sg.Slider((1, 50), default_value=10, orientation="h", size=(18, 10), key="GAIN"),
                sg.Text("10.00", key="GAIN_LABEL"),
                sg.Button("Auto-gain", key="AUTO_GAIN"),
                sg.Checkbox("Blink", key="BLINK"),
            ]
        ]

        image_row = [
            sg.Column([[sg.Text("Original")], [sg.Image(key="IMG_ORIG")]]),
            sg.Column([[sg.Text("Adversarial")], [sg.Image(key="IMG_ADV")]]),
            sg.Column([[sg.Text("Difference")], [sg.Image(key="IMG_DIFF")]]),
        ]

        layout = [
            [
                sg.Column(header_col + [[sg.Frame("Параметры", params_col, scrollable=True, vertical_scroll_only=True, size=(320, 380))]]),
                sg.Column(
                    [
                        [results_table],
                        [sg.Frame("Визуализация", preview_controls + [image_row], expand_x=True, expand_y=True)],
                    ],
                    expand_x=True,
                    expand_y=True,
                ),
            ],
            [sg.Frame("Запуск", run_col), sg.Frame("Журнал", log_col)],
        ]

        window = sg.Window("Атаки", layout, finalize=True, resizable=True)
        self.window = window
        self._refresh_visuals(window)

        while True:
            event, values = window.read(timeout=200)
            if event == sg.WIN_CLOSED:
                self.stop_flag.set()
                break

            if event == "SYNC":
                self._sync_from_main(window)
            elif event == "RUN":
                self._run_all(values, window)
            elif event == "STOP":
                self._stop_attacks(window)
            elif event == "RESULTS":
                if values["RESULTS"]:
                    self.selected_attack = ATTACK_ORDER[values["RESULTS"][0]]
                self._refresh_visuals(window)
            elif event in ("DIFF_SIGNED", "DIFF_ABS"):
                self.diff_mode = "signed" if values["DIFF_SIGNED"] else "abs"
                self._refresh_visuals(window)
            elif event == "GAIN":
                self.gain = float(values["GAIN"])
                window["GAIN_LABEL"].update(f"{self.gain:.2f}")
                self._refresh_visuals(window)
            elif event == "AUTO_GAIN":
                self._auto_gain(window)
            elif event == "BLINK":
                self._blink_on = values["BLINK"]
                self._blink_show_adv = False
            elif event == "ATTACK_PROGRESS":
                percent = int(values["ATTACK_PROGRESS"])
                window["ATTACK_PROGRESS"].update(f"Прогресс атаки: {percent}%")
                window["PROGRESS_BAR"].update(percent)
            elif event == "RESULT_UPDATE":
                name, res, done, total = values["RESULT_UPDATE"]
                self._update_result_row(window, name, res)
                if done is not None and total is not None:
                    window["PROGRESS"].update(f"Выполнение: {done}/{total}")
                self._refresh_visuals(window)
            elif event == "RUN_DONE":
                self.running = False
                window["RUN"].update(disabled=False)
                window["STOP"].update(disabled=True)
            elif event == "LOG":
                self._log(window, values["LOG"])

            if self._blink_on:
                self._blink_show_adv = not self._blink_show_adv
                self._refresh_visuals(window)

        window.close()

    def _log(self, window, text: str):
        current = window["LOG"].get()
        window["LOG"].update(current + text + "\n")

    def _sync_from_main(self, window):
        if self.app.train_images is None:
            self._log(window, "Датасет ещё не загружен.")
            return
        idx = self.app._get_active_index()
        if idx is None:
            self._log(window, "Выберите индекс в основном списке.")
            return
        self._load_index(idx, window)

    def _load_index(self, idx: int, window):
        self.current_idx = idx
        window["IDX"].update(f"{idx:05d}")
        self.current_y = int(self.app.train_labels[idx]) if self.app.train_labels is not None else None
        window["LABEL"].update(self._format_label(self.current_y))

        try:
            self.current_x0 = self.app._get_x01_from_index(idx)
            pred_text = self._predict_before_text(self.current_x0)
            window["PRED"].update(pred_text)
            self._reset_results(window)
            self._refresh_visuals(window)
        except Exception as exc:
            window["PRED"].update("-")
            self._log(window, f"Ошибка подготовки изображения: {exc}")
            self.current_x0 = None
            self._refresh_visuals(window)

    def _predict_before_text(self, x01: torch.Tensor) -> str:
        model_name = self.app.model_var.get().strip()
        if not model_name:
            return "-"
        model = self.app._get_model(model_name)
        norm_model = NormalizedModel(model, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        norm_model.eval()
        with torch.no_grad():
            logits = norm_model(x01)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            p = float(probs[0, pred].item())
        return self._format_pred(pred, p)

    def _reset_results(self, window):
        self.results = {}
        window["RESULTS"].update([[name, "-", "-", "-", "-", "-", "-", "-"] for name in ATTACK_ORDER])
        window["PROGRESS"].update("Выполнение: 0/6")
        window["ATTACK_PROGRESS"].update("Прогресс атаки: 0%")
        window["PROGRESS_BAR"].update(0)

    def _format_label(self, label: Optional[int]) -> str:
        if label is None:
            return "-"
        if 0 <= label < len(CLASSES):
            return f"{label} ({CLASSES[label]})"
        return str(label)

    def _format_pred(self, pred: int, p: float) -> str:
        return f"{pred} ({CLASSES[pred]}) (p={p:.3f})"

    def _run_all(self, values, window):
        if self.running:
            return
        if self.current_x0 is None or self.current_idx is None:
            self._log(window, "Сначала выберите изображение.")
            return
        if self.current_y is None:
            self._log(window, "Не удалось определить истинную метку.")
            return

        try:
            params = self._collect_params(values)
        except ValueError as exc:
            self._log(window, str(exc))
            return

        device = self._resolve_device(values, window)
        if device is None:
            return

        self.stop_flag.clear()
        self.running = True
        window["RUN"].update(disabled=True)
        window["STOP"].update(disabled=False)
        window["LOG"].update("")

        thread = threading.Thread(target=self._run_worker, args=(params, device, window), daemon=True)
        thread.start()

    def _stop_attacks(self, window):
        self.stop_flag.set()
        self._log(window, "Остановить: запрос на остановку отправлен.")

    def _resolve_device(self, values, window):
        choice = values["DEVICE"]
        if choice == "CUDA":
            if not torch.cuda.is_available():
                self._log(window, "CUDA недоступна на этом компьютере.")
                return None
            return torch.device("cuda")
        if choice == "CPU":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _collect_params(self, values):
        def fval(key, name, default=None):
            value = values[key].strip()
            if not value and default is not None:
                return default
            if not value:
                raise ValueError(f"Поле '{name}' не заполнено.")
            return float(value)

        def ival(key, name, default=None):
            value = values[key].strip()
            if not value and default is not None:
                return default
            if not value:
                raise ValueError(f"Поле '{name}' не заполнено.")
            return int(value)

        common_eps = fval("COMMON_EPS", "Эпсилон (общий)")
        time_limit = values["TIME_LIMIT"].strip()
        time_limit = float(time_limit) if time_limit else 0.0
        success_criteria = "pred" if values["CRIT_PRED"] else "true"

        params = {
            "epsilon": common_eps,
            "success_criteria": success_criteria,
            "time_limit": time_limit,
            "fgsm": {"epsilon": fval("FGSM_EPS", "FGSM epsilon", default=common_eps)},
            "bim": {
                "epsilon": fval("BIM_EPS", "BIM epsilon", default=common_eps),
                "alpha": fval("BIM_ALPHA", "BIM alpha"),
                "iters": ival("BIM_ITERS", "BIM iters"),
            },
            "pgd": {
                "epsilon": fval("PGD_EPS", "PGD epsilon", default=common_eps),
                "alpha": fval("PGD_ALPHA", "PGD alpha"),
                "iters": ival("PGD_ITERS", "PGD iters"),
                "random_start": bool(values["PGD_RANDOM"]),
            },
            "deepfool": {
                "max_iters": ival("DF_ITERS", "DeepFool max_iters"),
                "overshoot": fval("DF_OVERSHOOT", "DeepFool overshoot"),
            },
            "cw": {
                "steps": ival("CW_STEPS", "C&W steps"),
                "lr": fval("CW_LR", "C&W lr"),
                "c": fval("CW_C", "C&W c"),
                "kappa": fval("CW_KAPPA", "C&W kappa"),
                "binary_search_steps": ival("CW_BSS", "C&W binary_search_steps"),
            },
            "autoattack": {
                "epsilon": fval("AA_EPS", "AutoAttack epsilon", default=common_eps),
                "version": values["AA_VERSION"] or "standard",
            },
        }
        return params

    def _run_worker(self, params, device, window):
        try:
            model_name = self.app.model_var.get().strip()
            if not model_name:
                raise RuntimeError("Выберите модель в основном окне.")
            model = self.app._get_model(model_name)
            self._log(window, f"Модель: {model_name}")
            self._log(window, f"Устройство: {device.type}")

            total = len(ATTACK_ORDER)
            done = 0
            for name in ATTACK_ORDER:
                if self.stop_flag.is_set():
                    self._mark_remaining_stopped(name, window)
                    break

                self._log(window, f"Запуск атаки {name}...")
                window.write_event_value("ATTACK_PROGRESS", 0)
                res = run_attack(
                    name=name,
                    model=model,
                    x0=self.current_x0,
                    y_true=self.current_y,
                    params=params,
                    device=device,
                    stop_flag=self.stop_flag,
                    progress_cb=lambda n, c, t: window.write_event_value("ATTACK_PROGRESS", int((c / t) * 100)),
                )
                self.results[name] = res
                done += 1
                window.write_event_value("RESULT_UPDATE", (name, res, done, total))

                if res.get("stopped"):
                    self._mark_remaining_stopped(name, window)
                    break
        except Exception as exc:
            window.write_event_value("LOG", f"Ошибка выполнения: {exc}")
        finally:
            window.write_event_value("RUN_DONE", None)

    def _mark_remaining_stopped(self, current_name, window):
        if current_name not in ATTACK_ORDER:
            return
        idx = ATTACK_ORDER.index(current_name)
        for rest in ATTACK_ORDER[idx:]:
            if rest not in self.results:
                self.results[rest] = {"stopped": True}
                window.write_event_value("RESULT_UPDATE", (rest, self.results[rest], None, None))

    def _refresh_visuals(self, window):
        if not PIL_AVAILABLE:
            window["IMG_ORIG"].update(data=None)
            window["IMG_ADV"].update(data=None)
            window["IMG_DIFF"].update(data=None)
            return
        if self.current_x0 is None:
            window["IMG_ORIG"].update(data=None)
            window["IMG_ADV"].update(data=None)
            window["IMG_DIFF"].update(data=None)
            return

        if not self.selected_attack and ATTACK_ORDER:
            self.selected_attack = ATTACK_ORDER[0]
        res = self.results.get(self.selected_attack)
        adv = res.get("x_adv") if res else None
        diff_vis = self._compute_diff_vis(adv, self.current_x0) if adv is not None else None

        window["IMG_ORIG"].update(data=self._tensor_to_bytes(self.current_x0))
        if adv is not None:
            if self._blink_on and self._blink_show_adv:
                window["IMG_ADV"].update(data=self._tensor_to_bytes(self.current_x0))
            else:
                window["IMG_ADV"].update(data=self._tensor_to_bytes(adv))
        else:
            window["IMG_ADV"].update(data=None)
        window["IMG_DIFF"].update(data=self._tensor_to_bytes(diff_vis) if diff_vis is not None else None)

    def _compute_diff_vis(self, x_adv: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        gain = float(self.gain)
        if self.diff_mode == "signed":
            diff = x_adv - x0
            diff_vis = (0.5 + gain * diff).clamp(0.0, 1.0)
        else:
            diff = (x_adv - x0).abs()
            diff_vis = (gain * diff).clamp(0.0, 1.0)
        return diff_vis

    def _auto_gain(self, window):
        res = self.results.get(self.selected_attack)
        if not res or res.get("x_adv") is None or self.current_x0 is None:
            return
        diff = (res["x_adv"] - self.current_x0).abs()
        max_abs = float(diff.max().item())
        if max_abs <= 0:
            return
        gain = 0.4 / max_abs
        gain = max(1.0, min(50.0, gain))
        self.gain = gain
        window["GAIN"].update(value=gain)
        window["GAIN_LABEL"].update(f"{gain:.2f}")
        self._refresh_visuals(window)

    def _tensor_to_bytes(self, x01: torch.Tensor):
        if x01 is None:
            return None
        x = x01.detach().cpu().squeeze(0).clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        img = Image.fromarray(arr).resize((96, 96))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _on_close(self):
        self.stop_flag.set()

    def _update_result_row(self, window, name, res):
        pred_before = res.get("pred_before")
        p_before = res.get("p_before")
        before_text = self._format_pred(pred_before, p_before) if pred_before is not None else "-"

        pred_after = res.get("pred_after")
        p_after = res.get("p_after")
        after_text = self._format_pred(pred_after, p_after) if pred_after is not None else "-"

        success = res.get("success")
        success_text = "Да" if success else "Нет" if success is not None else "-"

        linf = res.get("linf")
        l2 = res.get("l2")
        linf_text = f"{linf:.4f}" if linf is not None else "-"
        l2_text = f"{l2:.4f}" if l2 is not None else "-"

        time_ms = res.get("time_ms")
        time_text = str(time_ms) if time_ms is not None else "-"

        status = "ОК"
        if res.get("stopped"):
            status = "Остановлено"
        if res.get("error"):
            status = "Ошибка"
            self._log(window, f"{name}: {res['error']}")
        if res.get("note"):
            self._log(window, f"{name}: {res['note']}")

        rows = window["RESULTS"].get()
        if rows:
            idx = ATTACK_ORDER.index(name)
            rows[idx] = [name, before_text, after_text, success_text, linf_text, l2_text, time_text, status]
            window["RESULTS"].update(rows)
        if self.selected_attack is None and res.get("x_adv") is not None:
            self.selected_attack = name
