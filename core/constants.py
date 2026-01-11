from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
BIN_DIR = PROJECT_ROOT / "stl10_binary"
TRAIN_X = BIN_DIR / "train_X.bin"
TRAIN_Y = BIN_DIR / "train_y.bin"

HEIGHT, WIDTH, DEPTH = 96, 96, 3

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
