import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, TYPE_CHECKING

import torch

from attacks import ATTACK_ORDER, NormalizedModel, run_attack
from core.constants import CLASSES
from core.transforms import PIL_AVAILABLE

if PIL_AVAILABLE:
    from PIL import Image, ImageTk

if TYPE_CHECKING:
    from ui.main_window import App


class AttacksWindow:
    def __init__(self, app: "App"):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Атаки (адверсариальные примеры)")
        self.window.geometry("1280x820")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.stop_flag = threading.Event()
        self.running = False
        self.current_idx = None
        self.current_x0 = None
        self.current_y = None
        self.results = {}
        self.result_items = {}

        self.orig_photo = None
        self.adv_photo = None
        self.diff_photo = None

        self.diff_mode_var = tk.StringVar(value="signed")
        self.gain_var = tk.DoubleVar(value=10.0)
        self.gain_label_var = tk.StringVar(value="10.00")
        self.blink_var = tk.BooleanVar(value=False)
        self._blink_after_id = None
        self._blink_showing_adv = False

        self.index_var = tk.StringVar(value="Индекс: -")
        self.label_var = tk.StringVar(value="Истинная метка: -")
        self.pred_before_var = tk.StringVar(value="Предсказание (до): -")
        self.progress_var = tk.StringVar(value="Выполнение: 0/6")
        self.attack_progress_var = tk.StringVar(value="Прогресс атаки: 0%")

        self.device_var = tk.StringVar(value="Авто")
        self.success_var = tk.StringVar(value="true")
        self.common_eps_var = tk.StringVar(value="0.0313725")
        self.time_limit_var = tk.StringVar(value="")

        self.fgsm_eps_var = tk.StringVar(value="")
        self.bim_eps_var = tk.StringVar(value="")
        self.bim_alpha_var = tk.StringVar(value="0.0078431")
        self.bim_iters_var = tk.StringVar(value="10")
        self.pgd_eps_var = tk.StringVar(value="")
        self.pgd_alpha_var = tk.StringVar(value="0.0078431")
        self.pgd_iters_var = tk.StringVar(value="20")
        self.pgd_random_var = tk.BooleanVar(value=True)
        self.df_iters_var = tk.StringVar(value="50")
        self.df_overshoot_var = tk.StringVar(value="0.02")
        self.cw_steps_var = tk.StringVar(value="200")
        self.cw_lr_var = tk.StringVar(value="0.01")
        self.cw_c_var = tk.StringVar(value="1.0")
        self.cw_kappa_var = tk.StringVar(value="0")
        self.cw_bss_var = tk.StringVar(value="5")
        self.aa_eps_var = tk.StringVar(value="")
        self.aa_version_var = tk.StringVar(value="standard")

        self._build_ui()
        self._refresh_visuals()

    def _build_ui(self):
        root = self.window
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, padding=10)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.rowconfigure(1, weight=1)
        sidebar.rowconfigure(3, weight=1)

        header = ttk.LabelFrame(sidebar, text="Выбранное изображение", padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.index_var).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.label_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, textvariable=self.pred_before_var).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="Синхронизировать с главным", command=self._sync_from_main).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

        params_wrap = ttk.LabelFrame(sidebar, text="Настройки атак", padding=6)
        params_wrap.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        params_wrap.rowconfigure(0, weight=1)
        params_wrap.columnconfigure(0, weight=1)

        params_canvas = tk.Canvas(params_wrap, highlightthickness=0)
        params_scroll = ttk.Scrollbar(params_wrap, orient=tk.VERTICAL, command=params_canvas.yview)
        params_canvas.configure(yscrollcommand=params_scroll.set)
        params_canvas.grid(row=0, column=0, sticky="nsew")
        params_scroll.grid(row=0, column=1, sticky="ns")

        params_frame = ttk.Frame(params_canvas)
        params_canvas.create_window((0, 0), window=params_frame, anchor="nw")

        def _on_params_configure(_event):
            params_canvas.configure(scrollregion=params_canvas.bbox("all"))

        params_frame.bind("<Configure>", _on_params_configure)

        self._build_general_params(params_frame)
        self._build_attack_params(params_frame)

        run_frame = ttk.LabelFrame(sidebar, text="Запуск", padding=10)
        run_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        run_frame.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            run_frame,
            text="Провести атаку на выбранное изображение",
            command=self._run_all,
        )
        self.run_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(run_frame, text="Остановить", command=self._stop_attacks, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(6, 0))
        ttk.Label(run_frame, textvariable=self.progress_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(run_frame, textvariable=self.attack_progress_var).grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.attack_progress = ttk.Progressbar(run_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.attack_progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        log_frame = ttk.LabelFrame(sidebar, text="Журнал", padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_results_table(right)
        self._build_preview(right)

    def _build_general_params(self, parent):
        general = ttk.LabelFrame(parent, text="Общие параметры", padding=8)
        general.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(general)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text="Устройство:").pack(side=tk.LEFT)
        device_combo = ttk.Combobox(row, textvariable=self.device_var, state="readonly", width=10)
        device_combo["values"] = ["Авто", "CPU", "CUDA"]
        device_combo.pack(side=tk.LEFT, padx=6)

        eps_row = ttk.Frame(general)
        eps_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(eps_row, text="Эпсилон (общий):").pack(side=tk.LEFT)
        ttk.Entry(eps_row, textvariable=self.common_eps_var, width=12).pack(side=tk.LEFT, padx=6)

        time_row = ttk.Frame(general)
        time_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(time_row, text="Лимит по времени (сек):").pack(side=tk.LEFT)
        ttk.Entry(time_row, textvariable=self.time_limit_var, width=8).pack(side=tk.LEFT, padx=6)

        crit = ttk.LabelFrame(general, text="Критерий успеха", padding=6)
        crit.pack(fill=tk.X, pady=(6, 0))
        ttk.Radiobutton(
            crit,
            text="Сменился класс относительно истинной метки",
            variable=self.success_var,
            value="true",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            crit,
            text="Сменился класс относительно предсказания до атаки",
            variable=self.success_var,
            value="pred",
        ).pack(anchor=tk.W)

    def _build_attack_params(self, parent):
        def entry_row(frame, label, var, width=8):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT, padx=6)

        fgsm = ttk.LabelFrame(parent, text="FGSM", padding=6)
        fgsm.pack(fill=tk.X, pady=4)
        entry_row(fgsm, "epsilon:", self.fgsm_eps_var)

        bim = ttk.LabelFrame(parent, text="BIM", padding=6)
        bim.pack(fill=tk.X, pady=4)
        entry_row(bim, "epsilon:", self.bim_eps_var)
        entry_row(bim, "alpha:", self.bim_alpha_var)
        entry_row(bim, "iters:", self.bim_iters_var)

        pgd = ttk.LabelFrame(parent, text="PGD", padding=6)
        pgd.pack(fill=tk.X, pady=4)
        entry_row(pgd, "epsilon:", self.pgd_eps_var)
        entry_row(pgd, "alpha:", self.pgd_alpha_var)
        entry_row(pgd, "iters:", self.pgd_iters_var)
        ttk.Checkbutton(pgd, text="Случайный старт", variable=self.pgd_random_var).pack(anchor=tk.W)

        deepfool = ttk.LabelFrame(parent, text="DeepFool", padding=6)
        deepfool.pack(fill=tk.X, pady=4)
        entry_row(deepfool, "max_iters:", self.df_iters_var)
        entry_row(deepfool, "overshoot:", self.df_overshoot_var)

        cw = ttk.LabelFrame(parent, text="C&W", padding=6)
        cw.pack(fill=tk.X, pady=4)
        entry_row(cw, "steps:", self.cw_steps_var)
        entry_row(cw, "lr:", self.cw_lr_var)
        entry_row(cw, "c:", self.cw_c_var)
        entry_row(cw, "kappa:", self.cw_kappa_var)
        entry_row(cw, "binary_search_steps:", self.cw_bss_var)

        aa = ttk.LabelFrame(parent, text="AutoAttack", padding=6)
        aa.pack(fill=tk.X, pady=4)
        entry_row(aa, "epsilon:", self.aa_eps_var)
        row = ttk.Frame(aa)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="version:").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.aa_version_var, state="readonly", width=10)
        combo["values"] = ["standard", "plus"]
        combo.pack(side=tk.LEFT, padx=6)

    def _build_results_table(self, parent):
        table_frame = ttk.LabelFrame(parent, text="Результаты атак", padding=6)
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("attack", "before", "after", "success", "linf", "l2", "time", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        self.tree.heading("attack", text="Атака")
        self.tree.heading("before", text="До: класс (p)")
        self.tree.heading("after", text="После: класс (p)")
        self.tree.heading("success", text="Успех")
        self.tree.heading("linf", text="Норма (Linf)")
        self.tree.heading("l2", text="Норма (L2)")
        self.tree.heading("time", text="Время, мс")
        self.tree.heading("status", text="Статус")

        self.tree.column("attack", width=90, anchor=tk.W)
        self.tree.column("before", width=180, anchor=tk.W)
        self.tree.column("after", width=180, anchor=tk.W)
        self.tree.column("success", width=70, anchor=tk.CENTER)
        self.tree.column("linf", width=90, anchor=tk.CENTER)
        self.tree.column("l2", width=90, anchor=tk.CENTER)
        self.tree.column("time", width=90, anchor=tk.CENTER)
        self.tree.column("status", width=110, anchor=tk.CENTER)

        for name in ATTACK_ORDER:
            item = self.tree.insert("", tk.END, values=(name, "-", "-", "-", "-", "-", "-", "-"))
            self.result_items[name] = item

        self.tree.bind("<<TreeviewSelect>>", self._on_result_select)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")

    def _build_preview(self, parent):
        preview = ttk.LabelFrame(parent, text="Визуализация", padding=8)
        preview.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        preview.rowconfigure(1, weight=1)
        preview.columnconfigure(0, weight=1)

        controls = ttk.Frame(preview)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Difference:").grid(row=0, column=0, sticky="w")
        self.diff_signed_rb = ttk.Radiobutton(
            controls,
            text="Signed",
            variable=self.diff_mode_var,
            value="signed",
            command=self._refresh_diff_image,
        )
        self.diff_signed_rb.grid(row=0, column=1, padx=(6, 0))
        self.diff_abs_rb = ttk.Radiobutton(
            controls,
            text="Abs",
            variable=self.diff_mode_var,
            value="abs",
            command=self._refresh_diff_image,
        )
        self.diff_abs_rb.grid(row=0, column=2, padx=(6, 12))

        ttk.Label(controls, text="Gain:").grid(row=0, column=3, sticky="w")
        self.gain_scale = ttk.Scale(
            controls,
            from_=1.0,
            to=50.0,
            orient=tk.HORIZONTAL,
            variable=self.gain_var,
            command=self._on_gain_change,
        )
        self.gain_scale.grid(row=0, column=4, sticky="ew", padx=(6, 6))
        self.gain_label = ttk.Label(controls, textvariable=self.gain_label_var, width=6)
        self.gain_label.grid(row=0, column=5, padx=(0, 8))

        self.auto_gain_button = ttk.Button(controls, text="Auto-gain", command=self._auto_gain)
        self.auto_gain_button.grid(row=0, column=6, padx=(0, 8))
        self.blink_button = ttk.Checkbutton(
            controls,
            text="Blink",
            variable=self.blink_var,
            command=self._toggle_blink,
        )
        self.blink_button.grid(row=0, column=7)

        panels = ttk.Frame(preview)
        panels.grid(row=1, column=0, sticky="nsew")
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.columnconfigure(2, weight=1)
        panels.rowconfigure(0, weight=1)

        orig_frame = ttk.LabelFrame(panels, text="Original", padding=6)
        orig_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.orig_label = tk.Label(orig_frame, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.orig_label.pack(pady=6, fill=tk.BOTH, expand=True)

        adv_frame = ttk.LabelFrame(panels, text="Adversarial", padding=6)
        adv_frame.grid(row=0, column=1, sticky="nsew", padx=6)

        self.adv_label = tk.Label(adv_frame, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.adv_label.pack(pady=6, fill=tk.BOTH, expand=True)

        diff_frame = ttk.LabelFrame(panels, text="Difference", padding=6)
        diff_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        self.diff_label = tk.Label(diff_frame, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.diff_label.pack(pady=6, fill=tk.BOTH, expand=True)

        self._preview_controls = [
            self.diff_signed_rb,
            self.diff_abs_rb,
            self.gain_scale,
            self.auto_gain_button,
            self.blink_button,
        ]

    def _log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _set_running(self, running: bool):
        self.running = running
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _sync_from_main(self):
        if self.app.train_images is None:
            messagebox.showerror("Ошибка", "Датасет ещё не загружен.")
            return
        idx = self.app._get_active_index()
        if idx is None:
            messagebox.showerror("Ошибка", "Выберите индекс в основном списке.")
            return
        self._load_index(idx)

    def _load_index(self, idx: int):
        self._stop_blink()
        self.current_idx = idx
        self.index_var.set(f"Индекс: {idx:05d}")
        self.current_y = int(self.app.train_labels[idx]) if self.app.train_labels is not None else None
        label_text = f"Истинная метка: {self._format_label(self.current_y)}"
        self.label_var.set(label_text)

        try:
            self.current_x0 = self.app._get_x01_from_index(idx)
            pred_text = self._predict_before_text(self.current_x0)
            self.pred_before_var.set(f"Предсказание (до): {pred_text}")
            self._reset_results(pred_text)
            self._refresh_visuals()
        except Exception as exc:
            self.pred_before_var.set("Предсказание (до): -")
            self._log(f"Ошибка подготовки изображения: {exc}")
            self.current_x0 = None
            self._refresh_visuals()

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

    def _reset_results(self, before_text: str):
        self.results = {}
        self.progress_var.set("Выполнение: 0/6")
        self.attack_progress_var.set("Прогресс атаки: 0%")
        self.attack_progress["value"] = 0
        for name in ATTACK_ORDER:
            item = self.result_items[name]
            self.tree.item(item, values=(name, before_text, "-", "-", "-", "-", "-", "-"))
        self.adv_label.configure(image="", text="")
        self.diff_label.configure(image="", text="")
        self.adv_photo = None
        self.diff_photo = None

    def _format_label(self, label: Optional[int]) -> str:
        if label is None:
            return "-"
        if 0 <= label < len(CLASSES):
            return f"{label} ({CLASSES[label]})"
        return str(label)

    def _format_pred(self, pred: int, p: float) -> str:
        return f"{pred} ({CLASSES[pred]}) (p={p:.3f})"

    def _run_all(self):
        if self.running:
            return
        if self.current_x0 is None or self.current_idx is None:
            messagebox.showerror("Ошибка", "Сначала выберите изображение.")
            return
        if self.current_y is None:
            messagebox.showerror("Ошибка", "Не удалось определить истинную метку.")
            return

        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        device = self._resolve_device()
        if device is None:
            return

        self.stop_flag.clear()
        self._set_running(True)
        self.log_text.delete("1.0", tk.END)
        thread = threading.Thread(target=self._run_worker, args=(params, device))
        thread.daemon = True
        thread.start()

    def _stop_attacks(self):
        self.stop_flag.set()
        self._log("Остановить: запрос на остановку отправлен.")

    def _resolve_device(self):
        choice = self.device_var.get()
        if choice == "CUDA":
            if not torch.cuda.is_available():
                messagebox.showerror("Ошибка", "CUDA недоступна на этом компьютере.")
                return None
            return torch.device("cuda")
        if choice == "CPU":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _collect_params(self):
        def fval(var, name, default=None):
            value = var.get().strip()
            if not value and default is not None:
                return default
            if not value:
                raise ValueError(f"Поле '{name}' не заполнено.")
            return float(value)

        def ival(var, name, default=None):
            value = var.get().strip()
            if not value and default is not None:
                return default
            if not value:
                raise ValueError(f"Поле '{name}' не заполнено.")
            return int(value)

        common_eps = fval(self.common_eps_var, "Эпсилон (общий)")
        time_limit = self.time_limit_var.get().strip()
        time_limit = float(time_limit) if time_limit else 0.0

        params = {
            "epsilon": common_eps,
            "success_criteria": self.success_var.get(),
            "time_limit": time_limit,
            "fgsm": {"epsilon": fval(self.fgsm_eps_var, "FGSM epsilon", default=common_eps)},
            "bim": {
                "epsilon": fval(self.bim_eps_var, "BIM epsilon", default=common_eps),
                "alpha": fval(self.bim_alpha_var, "BIM alpha"),
                "iters": ival(self.bim_iters_var, "BIM iters"),
            },
            "pgd": {
                "epsilon": fval(self.pgd_eps_var, "PGD epsilon", default=common_eps),
                "alpha": fval(self.pgd_alpha_var, "PGD alpha"),
                "iters": ival(self.pgd_iters_var, "PGD iters"),
                "random_start": bool(self.pgd_random_var.get()),
            },
            "deepfool": {
                "max_iters": ival(self.df_iters_var, "DeepFool max_iters"),
                "overshoot": fval(self.df_overshoot_var, "DeepFool overshoot"),
            },
            "cw": {
                "steps": ival(self.cw_steps_var, "C&W steps"),
                "lr": fval(self.cw_lr_var, "C&W lr"),
                "c": fval(self.cw_c_var, "C&W c"),
                "kappa": fval(self.cw_kappa_var, "C&W kappa"),
                "binary_search_steps": ival(self.cw_bss_var, "C&W binary_search_steps"),
            },
            "autoattack": {
                "epsilon": fval(self.aa_eps_var, "AutoAttack epsilon", default=common_eps),
                "version": self.aa_version_var.get().strip() or "standard",
            },
        }
        return params

    def _run_worker(self, params, device):
        try:
            model_name = self.app.model_var.get().strip()
            if not model_name:
                raise RuntimeError("Выберите модель в основном окне.")
            model = self.app._get_model(model_name)
            self._log(f"Модель: {model_name}")
            self._log(f"Устройство: {device.type}")

            total = len(ATTACK_ORDER)
            done = 0
            for name in ATTACK_ORDER:
                if self.stop_flag.is_set():
                    self._mark_remaining_stopped(name)
                    break

                self._log(f"Запуск атаки {name}...")
                self.window.after(0, self._set_attack_progress, 0)
                res = run_attack(
                    name=name,
                    model=model,
                    x0=self.current_x0,
                    y_true=self.current_y,
                    params=params,
                    device=device,
                    stop_flag=self.stop_flag,
                    progress_cb=self._attack_progress_cb,
                )
                self.results[name] = res
                done += 1
                self.window.after(0, self._update_result_row, name, res)
                self.window.after(0, self._update_progress, done, total)

                if res.get("stopped"):
                    self._mark_remaining_stopped(name)
                    break
        except Exception as exc:
            self.window.after(0, self._log, f"Ошибка выполнения: {exc}")
        finally:
            self.window.after(0, self._set_running, False)

    def _attack_progress_cb(self, name, current, total):
        if total <= 0:
            return
        percent = int((current / total) * 100)
        self.window.after(0, self._set_attack_progress, percent)

    def _set_attack_progress(self, percent: int):
        percent = max(0, min(100, percent))
        self.attack_progress_var.set(f"Прогресс атаки: {percent}%")
        self.attack_progress["value"] = percent

    def _mark_remaining_stopped(self, current_name):
        if current_name not in ATTACK_ORDER:
            return
        idx = ATTACK_ORDER.index(current_name)
        for rest in ATTACK_ORDER[idx:]:
            if rest not in self.results:
                self.results[rest] = {"stopped": True}
                self.window.after(0, self._update_result_row, rest, self.results[rest])

    def _update_progress(self, done, total):
        self.progress_var.set(f"Выполнение: {done}/{total}")

    def _update_result_row(self, name, res):
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
            self._log(f"{name}: {res['error']}")
        if res.get("note"):
            self._log(f"{name}: {res['note']}")

        item = self.result_items.get(name)
        if item:
            self.tree.item(item, values=(name, before_text, after_text, success_text, linf_text, l2_text, time_text, status))
            if not self.tree.selection():
                self.tree.selection_set(item)
                self.tree.see(item)

        if res.get("x_adv") is not None:
            self._refresh_visuals()

    def _on_result_select(self, _event):
        self._refresh_visuals()

    def _get_selected_attack_name(self):
        selection = self.tree.selection()
        if selection:
            return self.tree.item(selection[0], "values")[0]
        for name in ATTACK_ORDER:
            res = self.results.get(name)
            if res and res.get("x_adv") is not None:
                item = self.result_items.get(name)
                if item:
                    self.tree.selection_set(item)
                    self.tree.see(item)
                return name
        return ATTACK_ORDER[0] if ATTACK_ORDER else None

    def _get_selected_result(self):
        attack_name = self._get_selected_attack_name()
        if not attack_name:
            return None
        return self.results.get(attack_name)

    def _refresh_visuals(self):
        if not PIL_AVAILABLE:
            self._set_empty_preview(self.orig_label, "Preview requires Pillow")
            self._set_empty_preview(self.adv_label, "Preview requires Pillow")
            self._set_empty_preview(self.diff_label, "Preview requires Pillow")
            self._refresh_controls_state()
            self._stop_blink()
            return
        self._refresh_original_image()
        self._refresh_adv_image()
        self._refresh_diff_image()
        self._refresh_controls_state()
        self._refresh_blink_state()

    def _refresh_original_image(self):
        if self.current_x0 is None:
            self._set_empty_preview(self.orig_label, "Нет изображения")
            return
        self._set_original_preview(self.current_x0)

    def _refresh_adv_image(self):
        if self.current_x0 is None:
            self._set_empty_preview(self.adv_label, "Нет данных")
            return
        res = self._get_selected_result()
        if not res or res.get("x_adv") is None:
            self._set_empty_preview(self.adv_label, "Нет результата атаки")
            return
        if self.blink_var.get():
            if self._blink_showing_adv:
                self._set_adv_preview(res["x_adv"])
            else:
                self._set_adv_preview(self.current_x0)
            return
        self._set_adv_preview(res["x_adv"])

    def _refresh_diff_image(self):
        if self.current_x0 is None:
            self._set_empty_preview(self.diff_label, "Нет данных")
            return
        res = self._get_selected_result()
        if not res or res.get("x_adv") is None:
            self._set_empty_preview(self.diff_label, "Нет результата атаки")
            return
        diff_vis = self._compute_diff_vis(res["x_adv"], self.current_x0)
        self._set_diff_preview(diff_vis)

    def _refresh_controls_state(self):
        if not PIL_AVAILABLE or self.current_x0 is None:
            for widget in self._preview_controls:
                widget.configure(state=tk.DISABLED)
            return

        res = self._get_selected_result()
        has_adv = res is not None and res.get("x_adv") is not None
        for widget in self._preview_controls:
            widget.configure(state=tk.NORMAL if has_adv else tk.DISABLED)

    def _refresh_blink_state(self):
        if not self.blink_var.get():
            self._stop_blink()
            return
        if not self._can_blink():
            self.blink_var.set(False)
            self._stop_blink()
            return
        self._start_blink()

    def _can_blink(self):
        if not PIL_AVAILABLE:
            return False
        if self.current_x0 is None:
            return False
        res = self._get_selected_result()
        return bool(res and res.get("x_adv") is not None)

    def _set_empty_preview(self, label, text: str):
        label.configure(image="", text=text)
        if label is self.orig_label:
            self.orig_photo = None
        elif label is self.adv_label:
            self.adv_photo = None
        elif label is self.diff_label:
            self.diff_photo = None

    def _on_gain_change(self, value):
        try:
            gain = float(value)
        except ValueError:
            gain = float(self.gain_var.get())
        self.gain_label_var.set(f"{gain:.2f}")
        self._refresh_diff_image()

    def _auto_gain(self):
        res = self._get_selected_result()
        if not res or res.get("x_adv") is None or self.current_x0 is None:
            return
        diff = (res["x_adv"] - self.current_x0).abs()
        max_abs = float(diff.max().item())
        if max_abs <= 0:
            return
        gain = 0.4 / max_abs
        gain = max(1.0, min(50.0, gain))
        self.gain_var.set(gain)
        self.gain_label_var.set(f"{gain:.2f}")
        self._refresh_diff_image()

    def _toggle_blink(self):
        if self.blink_var.get():
            self._start_blink()
        else:
            self._stop_blink()
            self._refresh_adv_image()

    def _start_blink(self):
        self._stop_blink()
        if not self._can_blink():
            self.blink_var.set(False)
            return
        self._blink_showing_adv = False
        self._blink_tick()

    def _blink_tick(self):
        if not self.blink_var.get():
            return
        if not self._can_blink():
            self._stop_blink()
            self._refresh_adv_image()
            return
        res = self._get_selected_result()
        if self._blink_showing_adv:
            self._set_adv_preview(res["x_adv"])
        else:
            self._set_adv_preview(self.current_x0)
        self._blink_showing_adv = not self._blink_showing_adv
        self._blink_after_id = self.window.after(333, self._blink_tick)

    def _stop_blink(self):
        if self._blink_after_id is not None:
            self.window.after_cancel(self._blink_after_id)
            self._blink_after_id = None
        self._blink_showing_adv = False

    def _compute_diff_vis(self, x_adv: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        gain = float(self.gain_var.get())
        if self.diff_mode_var.get() == "signed":
            diff = x_adv - x0
            diff_vis = (0.5 + gain * diff).clamp(0.0, 1.0)
        else:
            diff = (x_adv - x0).abs()
            diff_vis = (gain * diff).clamp(0.0, 1.0)
        return diff_vis

    def _set_original_preview(self, x01: torch.Tensor):
        if not PIL_AVAILABLE:
            self.orig_label.configure(text="Preview requires Pillow", image="")
            return
        img = self._tensor_to_pil(x01)
        self.orig_photo = ImageTk.PhotoImage(img)
        self.orig_label.configure(image=self.orig_photo, text="")

    def _set_adv_preview(self, x01: torch.Tensor):
        if not PIL_AVAILABLE:
            self.adv_label.configure(text="Preview requires Pillow", image="")
            return
        img = self._tensor_to_pil(x01)
        self.adv_photo = ImageTk.PhotoImage(img)
        self.adv_label.configure(image=self.adv_photo, text="")

    def _set_diff_preview(self, x01: torch.Tensor):
        if not PIL_AVAILABLE:
            self.diff_label.configure(text="Preview requires Pillow", image="")
            return
        img = self._tensor_to_pil(x01)
        self.diff_photo = ImageTk.PhotoImage(img)
        self.diff_label.configure(image=self.diff_photo, text="")

    def _on_close(self):
        self.stop_flag.set()
        self._stop_blink()
        self.window.destroy()

    def _tensor_to_pil(self, x01: torch.Tensor):
        x = x01.detach().cpu().squeeze(0).clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        img = Image.fromarray(arr).rotate(-90, expand=True)
        img.thumbnail((256, 256))
        return img
