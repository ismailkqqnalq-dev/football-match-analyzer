"""
test_r1_detector.py
-------------------
Milestone M1 birim ve GPU entegrasyon testleri:
- ObjectDetector model.names hizalama doğrulaması (dict, list, COCO alias, mock)
- torch.inference_mode() altında detect_frame / detect_batch yürütmesi
- Ultralytics Boxes toplu .cpu().numpy() ayrıştırma, filtreleme ve boş sonuçlar
- CUDA OutOfMemoryError batch-halving + empty_cache() geri dönüşü ve batch=1 yeniden yükseltme
- utils/train_custom_yolo.py çevrimdışı yardımcı testleri (API key hijyeni, temp yollar,
  deterministik versiyon seçimi, data.yaml doğrulaması, save_dir ağırlık çözümü)
- RTX 4050 gerçek GPU çıkarım ve VRAM testi (ağırlık dosyası mevcutsa)
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from modules import detector as detector_module
from utils import train_custom_yolo


# ---------------------------------------------------------------------------
# Yardımcı Sahte Sınıflar
# ---------------------------------------------------------------------------
class _SpyTensorWrapper:
    """Toplu .cpu().numpy() çağrı sayısını doğrulamak için Tensor sarmalayıcısı."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.cpu_calls = 0

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        return self._tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 1. Config & ObjectDetector.__init__ model.names doğrulaması
# ---------------------------------------------------------------------------
def test_config_four_class_derived_constants():
    assert config.TARGET_CLASS_IDS == {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee",
    }
    assert config.BALL_CLASS_NAME == "ball"
    assert config.HUMAN_CLASS_NAMES == {"goalkeeper", "player", "referee"}
    assert config.USES_FINE_TUNED_FOOTBALL_MODEL is True
    assert config.TEAM_CLASSIFIER_N_CLUSTERS == 2


@pytest.mark.parametrize(
    "model_names,targets",
    [
        (
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        ),
        (
            ["ball", "goalkeeper", "player", "referee"],
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        ),
        (
            {0: "person", 32: "sports ball"},
            {0: "person", 32: "ball"},
        ),
    ],
)
def test_detector_validates_dict_list_and_coco_alias_names(monkeypatch, model_names, targets):
    class FakeYOLO:
        def __init__(self, model_path):
            self.names = model_names

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cpu",
        target_class_ids=targets,
    )
    assert det.target_class_ids == targets


@pytest.mark.parametrize(
    "bad_names",
    [
        {0: "player", 1: "goalkeeper", 2: "ball", 3: "referee"},
        {0: "person", 32: "sports ball"},
        ["ball", "goalkeeper"],
    ],
)
def test_detector_rejects_mismatched_model_names(monkeypatch, bad_names):
    class FakeYOLO:
        def __init__(self, model_path):
            self.names = bad_names

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    with pytest.raises(ValueError, match="TARGET_CLASS_IDS"):
        detector_module.ObjectDetector(
            model_path="dummy.pt",
            device="cpu",
            target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        )


def test_detector_allows_mock_without_names(monkeypatch):
    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cpu")
    assert det.target_class_ids == config.TARGET_CLASS_IDS


# ---------------------------------------------------------------------------
# 2. torch.inference_mode() yürütmesi
# ---------------------------------------------------------------------------
def test_detect_frame_and_batch_run_under_inference_mode(monkeypatch):
    observed_modes = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        def predict(self, **kwargs):
            observed_modes.append(torch.is_inference_mode_enabled())
            src = kwargs["source"]
            n = len(src) if isinstance(src, list) else 1
            empty = Boxes(torch.empty((0, 6), dtype=torch.float32), (64, 64))
            return [SimpleNamespace(boxes=empty) for _ in range(n)]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cpu",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    assert torch.is_inference_mode_enabled() is False
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect_frame(frame) == []
    assert det.detect_batch([frame, frame]) == [[], []]
    assert observed_modes == [True, True]
    assert det.detect_batch([]) == []
    assert len(observed_modes) == 2  # boş batch için predict çağrılmaz
    assert torch.is_inference_mode_enabled() is False


# ---------------------------------------------------------------------------
# 3. Toplu .cpu().numpy() ayrıştırma & boş kutu koruması
# ---------------------------------------------------------------------------
def test_bulk_boxes_parsing_parity_and_filtering(monkeypatch):
    raw = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 0.91234, 0],
            [50.0, 60.0, 80.0, 140.0, 0.88116, 2],
            [1.0, 2.0, 3.0, 4.0, 0.95, 99],
        ],
        dtype=torch.float32,
    )
    boxes = Boxes(raw, (480, 640))
    xyxy_spy = _SpyTensorWrapper(boxes.xyxy)
    cls_spy = _SpyTensorWrapper(boxes.cls)
    conf_spy = _SpyTensorWrapper(boxes.conf)

    class _SizedSpyBoxes:
        def __init__(self):
            self.xyxy = xyxy_spy
            self.cls = cls_spy
            self.conf = conf_spy

        def __len__(self):
            return len(raw)

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cpu",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    # Gerçek Ultralytics Boxes ile doğrula
    real_out = det._parse_result(SimpleNamespace(boxes=boxes))
    assert len(real_out) == 2
    assert real_out[0] == {
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "class_id": 0,
        "class_name": "ball",
        "confidence": 0.9123,
        "track_id": None,
        "team": None,
    }
    assert real_out[1] == {
        "bbox": [50.0, 60.0, 80.0, 140.0],
        "class_id": 2,
        "class_name": "player",
        "confidence": 0.8812,
        "track_id": None,
        "team": None,
    }
    assert all(isinstance(v, float) for item in real_out for v in item["bbox"])

    # Toplu dönüşümün satır sayısı yerine yalnızca 1 kez çağrıldığını doğrula
    spied_out = det._parse_result(SimpleNamespace(boxes=_SizedSpyBoxes()))
    assert spied_out == real_out
    assert xyxy_spy.cpu_calls == 1
    assert cls_spy.cpu_calls == 1
    assert conf_spy.cpu_calls == 1


def test_empty_boxes_and_none_results(monkeypatch):
    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        def predict(self, **kwargs):
            return []

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cpu",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    empty_boxes = Boxes(torch.empty((0, 6), dtype=torch.float32), (480, 640))
    assert det._parse_result(SimpleNamespace(boxes=empty_boxes)) == []
    assert det._parse_result(SimpleNamespace(boxes=None)) == []
    assert det._parse_result(None) == []
    assert det.detect_frame(np.zeros((32, 32, 3), dtype=np.uint8)) == []


# ---------------------------------------------------------------------------
# 4. CUDA OutOfMemoryError batch-halving & empty_cache()
# ---------------------------------------------------------------------------
def test_detect_batch_oom_halving_preserves_order(monkeypatch):
    call_sizes = []
    empty_cache_calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        def predict(self, **kwargs):
            src = kwargs["source"]
            call_sizes.append(len(src))
            if len(src) > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            results = []
            for frame in src:
                marker = float(frame[0, 0, 0])
                tensor = torch.tensor(
                    [[marker, 10.0, marker + 20.0, 40.0, 0.9, 2]],
                    dtype=torch.float32,
                )
                results.append(SimpleNamespace(boxes=Boxes(tensor, (64, 64))))
            return results

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        detector_module.torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cuda",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    frames4 = [np.full((32, 32, 3), i, dtype=np.uint8) for i in range(4)]
    out4 = det.detect_batch(frames4)
    assert call_sizes == [4, 2, 2]
    assert len(empty_cache_calls) == 1
    assert [d[0]["bbox"][0] for d in out4] == [0.0, 1.0, 2.0, 3.0]

    call_sizes.clear()
    empty_cache_calls.clear()
    frames5 = [np.full((32, 32, 3), i + 10, dtype=np.uint8) for i in range(5)]
    out5 = det.detect_batch(frames5)
    assert call_sizes == [5, 2, 3, 1, 2]
    assert len(empty_cache_calls) == 2
    assert [d[0]["bbox"][0] for d in out5] == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_detect_batch_oom_at_batch_size_one_reraises(monkeypatch):
    empty_cache_calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        def predict(self, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        detector_module.torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    det = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cuda",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        det.detect_batch([frame])
    assert len(empty_cache_calls) == 1

    # OOM olmayan RuntimeError empty_cache() çağırmadan doğrudan yükseltilmeli
    empty_cache_calls.clear()

    class FakeYOLONonOOM:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        def predict(self, **kwargs):
            raise RuntimeError("unexpected tensor shape mismatch")

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLONonOOM)
    det2 = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cuda",
        target_class_ids={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )
    with pytest.raises(RuntimeError, match="unexpected tensor shape mismatch"):
        det2.detect_batch([frame, frame])
    assert len(empty_cache_calls) == 0


# ---------------------------------------------------------------------------
# 5. utils/train_custom_yolo.py çevrimdışı yardımcı testleri
# ---------------------------------------------------------------------------
def test_train_script_api_key_env_priority_and_no_echo(monkeypatch, capsys):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "DUMMY_ENV_TOKEN")
    assert train_custom_yolo.resolve_api_key(None) == "DUMMY_ENV_TOKEN"

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError):
        train_custom_yolo.resolve_api_key(None)
    with pytest.raises(ValueError):
        train_custom_yolo.resolve_api_key("   ")

    masked = train_custom_yolo._sanitize_secret(
        "https://api.roboflow.com/?api_key=DUMMY_ENV_TOKEN&nocache=true",
        "DUMMY_ENV_TOKEN",
    )
    assert "DUMMY_ENV_TOKEN" not in masked
    assert "***" in masked

    captured = capsys.readouterr()
    assert "DUMMY_ENV_TOKEN" not in captured.out
    assert "DUMMY_ENV_TOKEN" not in captured.err


def test_train_script_temp_paths_outside_repo():
    root, dataset_dir, runs_dir = train_custom_yolo.resolve_temp_paths(
        None, "football-players-detection-3zvbc", "9"
    )
    temp_root = Path(tempfile.gettempdir()).resolve()
    repo_root = Path(config.BASE_DIR).resolve()

    assert root.is_relative_to(temp_root)
    assert dataset_dir.is_relative_to(root)
    assert runs_dir.is_relative_to(root)
    assert not dataset_dir.is_relative_to(repo_root)
    assert not runs_dir.is_relative_to(repo_root)


def test_train_script_deterministic_latest_version():
    requested = []

    class FakeProject:
        def versions(self):
            return [
                SimpleNamespace(version="2"),
                SimpleNamespace(version="12"),
                SimpleNamespace(version="9"),
                SimpleNamespace(version="5"),
            ]

        def version(self, ver_num):
            requested.append(ver_num)
            return SimpleNamespace(version=str(ver_num))

    proj = FakeProject()
    latest = train_custom_yolo.select_dataset_version(proj, None)
    assert latest.version == "12"
    explicit = train_custom_yolo.select_dataset_version(proj, 5)
    assert explicit.version == "5"
    assert requested == [12, 5]


def test_train_script_data_yaml_validation(tmp_path):
    yaml_list = tmp_path / "data_list.yaml"
    yaml_list.write_text(
        "names:\n  - ball\n  - goalkeeper\n  - player\n  - referee\n",
        encoding="utf-8",
    )
    assert train_custom_yolo.validate_data_yaml(yaml_list) == {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee",
    }

    yaml_dict = tmp_path / "data_dict.yaml"
    yaml_dict.write_text(
        "names:\n  0: ball\n  1: goalkeeper\n  2: player\n  3: referee\n",
        encoding="utf-8",
    )
    assert train_custom_yolo.validate_data_yaml(yaml_dict) == {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee",
    }

    yaml_bad = tmp_path / "data_bad.yaml"
    yaml_bad.write_text(
        "names:\n  - ball\n  - person\n  - referee\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        train_custom_yolo.validate_data_yaml(yaml_bad)


def test_train_script_save_dir_resolution_and_train_kwargs(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    fake_repo = tmp_path / "repo"
    (fake_repo / "utils").mkdir(parents=True)
    fake_script = fake_repo / "utils" / "train_custom_yolo.py"
    fake_script.write_text("# stub", encoding="utf-8")

    save_dir = work_dir / "runs" / "football_finetune2"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True)
    expected_best = weights_dir / "best.pt"
    expected_best.write_bytes(b"DUMMY_WEIGHTS_CONTENT")

    train_calls = []

    class FakeVersion:
        version = "3"

        def download(self, fmt, location=None, overwrite=False):
            loc = Path(location)
            loc.mkdir(parents=True, exist_ok=True)
            (loc / "data.yaml").write_text(
                "names:\n  - ball\n  - goalkeeper\n  - player\n  - referee\n",
                encoding="utf-8",
            )
            return SimpleNamespace(location=str(loc))

    class FakeProject:
        def versions(self):
            return [FakeVersion()]

        def version(self, ver_num):
            return FakeVersion()

    class FakeRoboflow:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def workspace(self, ws):
            return SimpleNamespace(project=lambda p: FakeProject())

    class FakeYOLO:
        def __init__(self, base_model):
            self.base_model = base_model
            self.trainer = SimpleNamespace(save_dir=str(save_dir), best=str(expected_best))

        def train(self, **kwargs):
            train_calls.append(kwargs)
            return SimpleNamespace()

    import roboflow
    import ultralytics

    monkeypatch.setenv("ROBOFLOW_API_KEY", "DUMMY_TEST_KEY")
    monkeypatch.setattr(train_custom_yolo, "__file__", str(fake_script))
    monkeypatch.setattr(roboflow, "Roboflow", FakeRoboflow)
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    train_custom_yolo.main(
        [
            "--work-dir",
            str(work_dir),
            "--epochs",
            "2",
            "--output-model-name",
            "test_best.pt",
        ]
    )

    assert len(train_calls) == 1
    kwargs = train_calls[0]
    assert kwargs["workers"] <= 2
    assert kwargs["cache"] is False
    assert kwargs["amp"] is True
    assert Path(kwargs["project"]).resolve() == (work_dir / "runs").resolve()

    copied = fake_repo / "models" / "test_best.pt"
    assert copied.is_file()
    assert copied.read_bytes() == b"DUMMY_WEIGHTS_CONTENT"


# ---------------------------------------------------------------------------
# 6. Gerçek RTX 4050 GPU çıkarım ve VRAM testi
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(config.YOLO_MODEL_PATH),
    reason="CUDA GPU veya eğitilmiş 4 sınıflı model ağırlığı mevcut değil",
)
def test_rtx4050_gpu_detector_inference_and_vram():
    det = detector_module.ObjectDetector(device="cuda", half=True)
    assert det.device == "cuda"
    assert det.half is True
    assert set(det.model.names.values()) == {"ball", "goalkeeper", "player", "referee"}

    torch.cuda.reset_peak_memory_stats()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(config.DETECTION_BATCH_SIZE)]
    single_out = det.detect_frame(frames[0])
    batch_out = det.detect_batch(frames)
    torch.cuda.synchronize()

    assert isinstance(single_out, list)
    assert len(batch_out) == config.DETECTION_BATCH_SIZE
    for frame_dets in batch_out:
        assert isinstance(frame_dets, list)
        for item in frame_dets:
            assert set(item.keys()) == {"bbox", "class_id", "class_name", "confidence", "track_id", "team"}
            assert item["class_name"] in {"ball", "goalkeeper", "player", "referee"}

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    assert peak_vram_mb < 4096.0
