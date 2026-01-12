import numpy as np

from core.constants import DEPTH, HEIGHT, TRAIN_X, TRAIN_Y, WIDTH


def load_train_dataset():
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

    return images, labels
