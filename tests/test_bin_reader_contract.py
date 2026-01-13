from pathlib import Path

import pytest

from na_core.constants import BIN_DIR
from na_core.dataset import load_train_dataset


def test_bin_reader_contract():
    train_x = Path(BIN_DIR) / "train_X.bin"
    train_y = Path(BIN_DIR) / "train_y.bin"
    if not train_x.exists() or not train_y.exists():
        pytest.skip("Dataset binaries not available")
    images, labels = load_train_dataset()
    assert images.dtype == "uint8"
    assert images.ndim == 4
    assert images.shape[1:] == (3, 96, 96)
    assert labels.dtype == "int64"
    assert len(labels) == images.shape[0]
