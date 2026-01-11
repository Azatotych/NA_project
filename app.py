import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from attacks import ATTACK_ORDER, NormalizedModel, run_attack


CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

MODEL_DIR = Path("models")
BIN_DIR = Path("stl10_binary")
TRAIN_X = BIN_DIR / "train_X.bin"
TRAIN_Y = BIN_DIR / "train_y.bin"
HEIGHT, WIDTH, DEPTH = 96, 96, 3


def build_model():
    try:
        model = models.vgg16(weights=None)
    except TypeError:
        model = models.vgg16(pretrained=False)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 10),
    )
    return model


PIL_AVAILABLE = False
try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


PIL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def preprocess(img):
    if PIL_AVAILABLE:
        return PIL_TRANSFORM(img)

    if not isinstance(img, torch.Tensor):
        raise TypeError("Unexpected image type without PIL")

    x = img.float().div(255.0)
    x = TF.resize(x, 256, antialias=True)
    x = TF.center_crop(x, [224, 224])

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x


ATTACK_TFM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def preprocess_x01(img):
    if PIL_AVAILABLE:
        return ATTACK_TFM(img)

    if not isinstance(img, torch.Tensor):
        raise TypeError("Unexpected image type without PIL")

    x = img.float().div(255.0)
    x = TF.resize(x, 256, antialias=True)
    x = TF.center_crop(x, [224, 224])
    return x


def load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("STL10 VGG16 Inference")
        self.root.geometry("900x600")

        self.model_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.selected_images = []
        self.model_cache = {}
        self.preview_image = None
        self.preview_path = None
        self.train_images = None
        self.train_labels = None
        self.pred_cache = {}
        self.attacks_window = None

        self._build_ui()
        self._refresh_models()
        self._load_dataset_async()

    def _build_ui(self):
        root = self.root

        top = ttk.Frame(root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        ttk.Button(top, text="Refresh", command=self._refresh_models).pack(side=tk.LEFT, padx=6)

        mid = ttk.Frame(root, padding=10)
        mid.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Train images:").pack(anchor=tk.W)
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.image_list = tk.Listbox(list_frame, height=16, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=scrollbar.set)

        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list.bind("<<ListboxSelect>>", self._on_list_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Clear selection", command=self._clear_images).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Run", command=self._run_inference).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Атаки:", command=self._open_attacks_window).pack(side=tk.LEFT, padx=6)

        right = ttk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        preview = ttk.Frame(right)
        preview.pack(fill=tk.X, expand=False)
        ttk.Label(preview, text="Preview:").pack(anchor=tk.W)
        self.preview_label = tk.Label(preview, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.preview_label.pack(pady=6, fill=tk.X)
        self.preview_info = ttk.Label(preview, text="No image selected")
        self.preview_info.pack(anchor=tk.W)

        ttk.Label(right, text="Output:").pack(anchor=tk.W, pady=(8, 0))
        self.output = tk.Text(right, height=18, wrap=tk.NONE)
        self.output.pack(fill=tk.BOTH, expand=True, pady=6)

        bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor=tk.W)

    def _refresh_models(self):
        models = sorted(MODEL_DIR.glob("*.pt"))
        names = [p.name for p in models]
        self.model_combo["values"] = names
        if names:
            current = self.model_var.get()
            if current not in names:
                self.model_var.set(names[0])
        else:
            self.model_var.set("")
        self.status_var.set("Models refreshed")

    def _load_dataset_async(self):
        self.status_var.set("Loading STL10 train dataset...")
        thread = threading.Thread(target=self._load_dataset)
        thread.daemon = True
        thread.start()

    def _load_dataset(self):
        try:
            if not TRAIN_X.exists() or not TRAIN_Y.exists():
                raise FileNotFoundError("Missing stl10_binary/train_X.bin or train_y.bin")

            images = np.fromfile(str(TRAIN_X), dtype=np.uint8)
            labels = np.fromfile(str(TRAIN_Y), dtype=np.uint8)

            if images.size % (HEIGHT * WIDTH * DEPTH) != 0:
                raise ValueError("Bad train_X.bin size")

            count = images.size // (HEIGHT * WIDTH * DEPTH)
            images = images.reshape(count, DEPTH, HEIGHT, WIDTH)
            labels = labels.astype(np.int64) - 1

            if len(labels) != count:
                raise ValueError("Labels/images mismatch")

            def update():
                self.train_images = images
                self.train_labels = labels
                self.image_list.delete(0, tk.END)
                for i in range(count):
                    self.image_list.insert(tk.END, f"{i:05d}")
                self.status_var.set(f"Loaded train set: {count} images")

            self.root.after(0, update)
        except Exception as exc:
            def update_error():
                self.status_var.set("Dataset load failed")
                messagebox.showerror("Dataset load failed", str(exc))

            self.root.after(0, update_error)

    def _clear_images(self):
        self.selected_images = []
        self.image_list.selection_clear(0, tk.END)
        self.status_var.set("Selection cleared")
        self._clear_preview()

    def _run_inference(self):
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showerror("No model", "Select a model first.")
            return
        if not self.selected_images:
            messagebox.showerror("No images", "Select images first.")
            return

        self.output.delete("1.0", tk.END)
        self.status_var.set("Running inference...")

        thread = threading.Thread(target=self._worker, args=(model_name, list(self.selected_images)))
        thread.daemon = True
        thread.start()

    def _on_list_select(self, _event):
        selection = self.image_list.curselection()
        if not selection:
            return
        indices = list(selection)
        self.selected_images = indices
        idx = indices[0]
        self.status_var.set(f"Selected {len(indices)} images")
        if self.train_images is not None and 0 <= idx < len(self.train_images):
            self._update_preview_index(idx)

    def _open_attacks_window(self):
        if self.attacks_window and self.attacks_window.window.winfo_exists():
            self.attacks_window.window.lift()
            return
        self.attacks_window = AttacksWindow(self)

    def _get_active_index(self):
        selection = self.image_list.curselection()
        if not selection:
            return None
        active = self.image_list.index(tk.ACTIVE)
        if active in selection:
            return active
        return selection[-1]

    def _worker(self, model_name, images):
        try:
            model = self._get_model(model_name)
            lines = []
            batch = []
            indices = list(images)
            for idx in indices:
                batch.append(self._get_tensor_from_index(idx))

            x = torch.stack(batch, dim=0)
            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=1).tolist()

            for idx, pred in zip(indices, preds):
                self.pred_cache[idx] = f"{pred} ({CLASSES[pred]})"
                label = self._label_name(idx)
                lines.append(f"{idx:05d}: pred {pred} ({CLASSES[pred]}) | label {label}")

            if indices:
                self._update_preview_index(indices[0])

            self._set_output(lines, f"Done. Processed {len(indices)} images.")
        except Exception as exc:
            self._set_output([f"Error: {exc}"], "Error")

    def _set_output(self, lines, status):
        def update():
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "\n".join(lines))
            self.status_var.set(status)

        self.root.after(0, update)

    def _clear_preview(self):
        self.preview_image = None
        self.preview_path = None
        self.preview_label.configure(image="", text="")
        self.preview_info.configure(text="No image selected")

    def _update_preview_index(self, idx: int):
        if self.train_images is None:
            return
        if not PIL_AVAILABLE:
            self.preview_label.configure(image="", text="Preview requires Pillow")
            self.preview_info.configure(text=f"Index {idx:05d}")
            return

        try:
            arr = self.train_images[idx]
            img = Image.fromarray(arr.transpose(1, 2, 0)).rotate(-90, expand=True)
            w, h = img.size
            preview = img.copy()
            preview.thumbnail((256, 256))
            self.preview_image = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_image, text="")
            label = self._label_name(idx)
            pred = self.pred_cache.get(idx)
            pred_text = f" | pred: {pred}" if pred is not None else ""
            self.preview_info.configure(text=f"Index {idx:05d} ({w}x{h}) | label: {label}{pred_text}")
            self.preview_path = None
        except Exception as exc:
            self.preview_label.configure(image="", text="Preview failed")
            self.preview_info.configure(text=f"Index {idx:05d} ({exc})")

    def _label_name(self, idx: int):
        if self.train_labels is None:
            return "unknown"
        label = int(self.train_labels[idx])
        if 0 <= label < len(CLASSES):
            return f"{label} ({CLASSES[label]})"
        return str(label)

    def _get_model(self, model_name):
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = build_model()
        state = load_state_dict(model_path)
        model.load_state_dict(state)
        model.eval()

        self.model_cache = {model_name: model}
        return model

    def _get_x01_from_index(self, idx: int):
        arr = self.train_images[idx]
        if PIL_AVAILABLE:
            img = Image.fromarray(arr.transpose(1, 2, 0))
            x01 = preprocess_x01(img)
        else:
            x = torch.from_numpy(arr)
            x01 = preprocess_x01(x)
        return x01.unsqueeze(0)

    def _get_tensor_from_index(self, idx: int):
        arr = self.train_images[idx]
        if PIL_AVAILABLE:
            img = Image.fromarray(arr.transpose(1, 2, 0))
            return preprocess(img)

        x = torch.from_numpy(arr).float().div(255.0)
        x = TF.resize(x, 256, antialias=True)
        x = TF.center_crop(x, [224, 224])
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return x


class AttacksWindow:
    def __init__(self, app: App):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Атаки (адверсариальные примеры)")
        self.window.geometry("1200x800")

        self.stop_flag = threading.Event()
        self.running = False
        self.current_idx = None
        self.current_x0 = None
        self.current_y = None
        self.results = {}
        self.result_items = {}
        self.orig_photo = None
        self.after_photo = None

        self.index_var = tk.StringVar(value="Индекс: -")
        self.label_var = tk.StringVar(value="Истинная метка: -")
        self.pred_before_var = tk.StringVar(value="Предсказание (до): -")
        self.progress_var = tk.StringVar(value="Выполнение: 0/6")
        self.attack_progress_var = tk.StringVar(value="Прогресс атаки: 0%")

        self.device_var = tk.StringVar(value="Авто")
        self.success_var = tk.StringVar(value="true")
        self.diff_var = tk.BooleanVar(value=False)

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

    def _build_ui(self):
        root = self.window
        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        left_container = ttk.Frame(main)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        left_canvas = tk.Canvas(left_container, highlightthickness=0, width=420)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        left.bind("<Configure>", _on_left_configure)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        selected = ttk.LabelFrame(left, text="Выбранное изображение", padding=10)
        selected.pack(fill=tk.X, expand=False)

        ttk.Label(selected, textvariable=self.index_var).pack(anchor=tk.W)
        ttk.Button(selected, text="Обновить из основного окна", command=self._sync_from_main).pack(
            anchor=tk.W, pady=6
        )
        ttk.Label(selected, textvariable=self.label_var).pack(anchor=tk.W)
        ttk.Label(selected, textvariable=self.pred_before_var).pack(anchor=tk.W)

        params = ttk.LabelFrame(left, text="Параметры атак", padding=10)
        params.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self._build_general_params(params)
        self._build_attack_params(params)

        run_block = ttk.LabelFrame(left, text="Запуск", padding=10)
        run_block.pack(fill=tk.X, pady=(10, 0))

        self.run_button = ttk.Button(
            run_block,
            text="Провести атаку на выбранное изображение",
            command=self._run_all,
        )
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(run_block, text="Остановить", command=self._stop_attacks, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)
        ttk.Label(run_block, textvariable=self.progress_var).pack(side=tk.LEFT, padx=6)
        ttk.Label(run_block, textvariable=self.attack_progress_var).pack(side=tk.LEFT, padx=6)

        self.attack_progress = ttk.Progressbar(run_block, orient=tk.HORIZONTAL, length=160, mode="determinate")
        self.attack_progress.pack(side=tk.LEFT, padx=6)

        log_frame = ttk.LabelFrame(left, text="Журнал", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        results = ttk.LabelFrame(right, text="Результаты", padding=10)
        results.pack(fill=tk.BOTH, expand=True)

        self._build_results_table(results)
        self._build_preview(results)

    def _build_general_params(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text="Устройство:").pack(side=tk.LEFT)
        device_combo = ttk.Combobox(row, textvariable=self.device_var, state="readonly", width=10)
        device_combo["values"] = ["Авто", "CPU", "CUDA"]
        device_combo.pack(side=tk.LEFT, padx=6)

        eps_row = ttk.Frame(parent)
        eps_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(eps_row, text="Эпсилон (общий):").pack(side=tk.LEFT)
        ttk.Entry(eps_row, textvariable=self.common_eps_var, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Label(eps_row, text="Ограничение по времени на атаку (сек):").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(eps_row, textvariable=self.time_limit_var, width=8).pack(side=tk.LEFT, padx=6)

        crit = ttk.Frame(parent)
        crit.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(crit, text="Критерий успеха:").pack(anchor=tk.W)
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
        columns = ("attack", "before", "after", "success", "linf", "l2", "time", "status")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
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

        tree_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_preview(self, parent):
        preview = ttk.Frame(parent)
        preview.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left = ttk.LabelFrame(preview, text="Оригинал", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.orig_label = tk.Label(left, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.orig_label.pack(pady=6, fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(preview, text="После атаки", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.after_label = tk.Label(right, anchor=tk.CENTER, background="#222", width=256, height=256)
        self.after_label.pack(pady=6, fill=tk.BOTH, expand=True)

        ttk.Checkbutton(right, text="Показать |Δ|", variable=self.diff_var, command=self._refresh_after_image).pack(
            anchor=tk.W
        )

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
        self.current_idx = idx
        self.index_var.set(f"Индекс: {idx:05d}")
        self.current_y = int(self.app.train_labels[idx]) if self.app.train_labels is not None else None
        label_text = f"Истинная метка: {self._format_label(self.current_y)}"
        self.label_var.set(label_text)

        try:
            self.current_x0 = self.app._get_x01_from_index(idx)
            self._set_original_preview(self.current_x0)
            pred_text = self._predict_before_text(self.current_x0)
            self.pred_before_var.set(f"Предсказание (до): {pred_text}")
            self._reset_results(pred_text)
        except Exception as exc:
            self.pred_before_var.set("Предсказание (до): -")
            self._log(f"Ошибка подготовки изображения: {exc}")

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
        self.after_label.configure(image="", text="")
        self.after_photo = None

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

        if res.get("x_adv") is not None:
            self._refresh_after_image()

    def _on_result_select(self, _event):
        self._refresh_after_image()

    def _refresh_after_image(self):
        selection = self.tree.selection()
        if not selection:
            return
        attack_name = self.tree.item(selection[0], "values")[0]
        res = self.results.get(attack_name)
        if not res or res.get("x_adv") is None or self.current_x0 is None:
            return
        x_adv = res["x_adv"]
        if self.diff_var.get():
            x_show = (x_adv - self.current_x0).abs().clamp(0.0, 1.0)
        else:
            x_show = x_adv
        self._set_after_preview(x_show)

    def _set_original_preview(self, x01: torch.Tensor):
        if not PIL_AVAILABLE:
            self.orig_label.configure(text="Preview requires Pillow", image="")
            return
        img = self._tensor_to_pil(x01)
        self.orig_photo = ImageTk.PhotoImage(img)
        self.orig_label.configure(image=self.orig_photo, text="")

    def _set_after_preview(self, x01: torch.Tensor):
        if not PIL_AVAILABLE:
            self.after_label.configure(text="Preview requires Pillow", image="")
            return
        img = self._tensor_to_pil(x01)
        self.after_photo = ImageTk.PhotoImage(img)
        self.after_label.configure(image=self.after_photo, text="")

    def _tensor_to_pil(self, x01: torch.Tensor):
        x = x01.detach().cpu().squeeze(0).clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        img = Image.fromarray(arr).rotate(-90, expand=True)
        img.thumbnail((256, 256))
        return img


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
