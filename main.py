import importlib
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

TTKBOOTSTRAP_AVAILABLE = False
try:
    import ttkbootstrap as ttkbootstrap

    TTKBOOTSTRAP_AVAILABLE = True
except Exception:
    TTKBOOTSTRAP_AVAILABLE = False

from ui.main_window import App
from core.constants import BIN_DIR, MODEL_DIR
from core.settings import load_settings, save_settings


def _check_libraries():
    required = ["torch", "torchvision"]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _select_data_dir(parent, current_dir):
    initial_dir = current_dir or str(BIN_DIR)
    return filedialog.askdirectory(
        parent=parent,
        title="Выберите источник данных (stl10_binary)",
        initialdir=initial_dir,
    )


def _select_model_path(parent, current_path):
    initial_dir = current_path or str(MODEL_DIR)
    return filedialog.askopenfilename(
        parent=parent,
        title="Выберите модель (.pt)",
        initialdir=initial_dir,
        filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")],
    )


def main():
    check_root = tk.Tk()
    check_root.withdraw()
    missing = _check_libraries()
    if missing:
        messagebox.showerror(
            "Отсутствуют библиотеки",
            "Не найдены необходимые библиотеки: " + ", ".join(missing),
            parent=check_root,
        )
        check_root.destroy()
        sys.exit(1)

    settings = load_settings()
    data_dir = settings.get("data_dir")
    model_path = settings.get("model_path")

    if not data_dir:
        data_dir = _select_data_dir(check_root, data_dir)
    if not model_path:
        model_path = _select_model_path(check_root, model_path)

    if not data_dir or not model_path:
        check_root.destroy()
        sys.exit(0)

    settings["data_dir"] = data_dir
    settings["model_path"] = model_path
    save_settings(settings)
    check_root.destroy()

    if TTKBOOTSTRAP_AVAILABLE:
        root = ttkbootstrap.Window(themename="darkly")
    else:
        root = tk.Tk()
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

    app = App(root, data_dir=data_dir, model_path=model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
