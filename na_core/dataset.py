from pathlib import Path

import numpy as np

from na_core.constants import BIN_DIR, DEPTH, HEIGHT, WIDTH


def load_train_dataset(data_dir=None):
    base_dir = Path(data_dir) if data_dir else BIN_DIR
    train_x = base_dir / "train_X.bin"
    train_y = base_dir / "train_y.bin"
    if not train_x.exists() or not train_y.exists():
        raise FileNotFoundError("Missing stl10_binary/train_X.bin or train_y.bin")

    images = np.fromfile(str(train_x), dtype=np.uint8)
    labels = np.fromfile(str(train_y), dtype=np.uint8)

    if images.size % (HEIGHT * WIDTH * DEPTH) != 0:
        raise ValueError("Bad train_X.bin size")

    count = images.size // (HEIGHT * WIDTH * DEPTH)
    images = images.reshape(count, DEPTH, HEIGHT, WIDTH)
    labels = labels.astype(np.int64) - 1

    if len(labels) != count:
        raise ValueError("Labels/images mismatch")

    return images, labels
