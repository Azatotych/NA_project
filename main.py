import tkinter as tk
from tkinter import ttk

TTKBOOTSTRAP_AVAILABLE = False
try:
    import ttkbootstrap as ttkbootstrap

    TTKBOOTSTRAP_AVAILABLE = True
except Exception:
    TTKBOOTSTRAP_AVAILABLE = False

from ui.main_window import App


def main():
    if TTKBOOTSTRAP_AVAILABLE:
        root = ttkbootstrap.Window(themename="darkly")
    else:
        root = tk.Tk()
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
