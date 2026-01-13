from pathlib import Path

import pytest
import torch

from na_core.constants import MODEL_DIR
from na_core.model_io import build_model, load_state_dict


def test_model_loader_contract():
    model_files = sorted(Path(MODEL_DIR).glob("*.pt"))
    if not model_files:
        pytest.skip("No model weights available")
    model = build_model()
    state = load_state_dict(model_files[0])
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10)
