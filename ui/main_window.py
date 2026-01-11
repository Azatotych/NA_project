import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torchvision.transforms.functional as TF

from core.constants import BIN_DIR, CLASSES, MODEL_DIR
from core.dataset import load_train_dataset
from core.model_io import build_model, load_state_dict
from core.settings import load_settings, save_settings
from core.transforms import PIL_AVAILABLE, preprocess, preprocess_x01
from ui.attacks_window import AttacksWindow

if PIL_AVAILABLE:
    from PIL import Image, ImageTk


class App:
    def __init__(self, root, data_dir=None, model_path=None):
        self.root = root
        self.root.title("STL10 VGG16 Inference")
        self.root.geometry("900x600")

        self.model_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.data_dir = data_dir
        self.model_path = model_path
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
        ttk.Button(top, text="Data source...", command=self._change_data_source).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Model...", command=self._change_model).pack(side=tk.LEFT, padx=6)

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
        if self.model_path:
            model_path = Path(self.model_path)
            names = [model_path.name] if model_path.exists() else []
        else:
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
            images, labels = load_train_dataset(self.data_dir)

            def update():
                self.train_images = images
                self.train_labels = labels
                self.image_list.delete(0, tk.END)
                for i in range(len(labels)):
                    self.image_list.insert(tk.END, f"{i:05d}")
                self.status_var.set(f"Loaded train set: {len(labels)} images")

            self.root.after(0, update)
        except Exception as exc:

            def update_error():
                self.status_var.set("Dataset load failed")
                messagebox.showerror("Dataset load failed", str(exc))

        self.root.after(0, update_error)

    def _change_data_source(self):
        initial_dir = self.data_dir or str(BIN_DIR)
        new_dir = filedialog.askdirectory(
            parent=self.root,
            title="Выберите источник данных (stl10_binary)",
            initialdir=initial_dir,
        )
        if not new_dir:
            return
        self.data_dir = new_dir
        settings = load_settings()
        settings["data_dir"] = new_dir
        save_settings(settings)
        self._load_dataset_async()

    def _change_model(self):
        initial_dir = self.model_path or str(MODEL_DIR)
        new_model = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите модель (.pt)",
            initialdir=initial_dir,
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")],
        )
        if not new_model:
            return
        self.model_path = new_model
        self.model_cache.clear()
        settings = load_settings()
        settings["model_path"] = new_model
        save_settings(settings)
        self._refresh_models()

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

        if self.model_path:
            model_path = Path(self.model_path)
        else:
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
