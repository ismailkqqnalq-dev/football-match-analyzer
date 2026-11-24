"""
Pytest ortak kanca ve işaret (marker) yönetimi.
Model ağırlığı veya GPU olmayan ortamlarda (ör. CI) ilgili testleri açık
gerekçeyle atlar.
"""

from pathlib import Path
import sys

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import config  # noqa: E402


MODEL_MISSING_SKIP_REASON = (
    "Model dosyası yok: models/football_yolo11s_best.pt – README 'Model ağırlıkları' bölümüne bakın"
)
PITCH_MODEL_MISSING_SKIP_REASON = (
    "Saha anahtar nokta modeli yok: models/football_pitch_pose.pt – README 'Model ağırlıkları' bölümüne bakın"
)
GPU_MISSING_SKIP_REASON = "CUDA GPU mevcut değil; GPU testi atlandı."


def pytest_runtest_setup(item):
    """requires_model, requires_pitch_model ve gpu işaretli testleri ortam eksikse anlamlı mesajla atlar."""
    if item.get_closest_marker("requires_model") is not None:
        model_path = Path(config.YOLO_MODEL_PATH)
        if not model_path.is_file():
            pytest.skip(MODEL_MISSING_SKIP_REASON)
    if item.get_closest_marker("requires_pitch_model") is not None:
        pitch_model_path = Path(config.PITCH_KEYPOINT_MODEL_PATH)
        if not pitch_model_path.is_file():
            pytest.skip(PITCH_MODEL_MISSING_SKIP_REASON)
    if item.get_closest_marker("gpu") is not None:
        if not torch.cuda.is_available():
            pytest.skip(GPU_MISSING_SKIP_REASON)
