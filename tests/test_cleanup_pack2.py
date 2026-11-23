import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


# ----------------------------------------------------------------------
# R4 KABUL TESTLERİ: Model İndirmede Zorunlu SHA-256 Kontrolü
# ----------------------------------------------------------------------
def test_r4_download_models_rejects_missing_sha256(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    called = {"urlopen": 0}

    def _fail_urlopen(*args, **kwargs):
        called["urlopen"] += 1
        raise AssertionError("SHA-256 olmadan indirme denenmemeliydi!")

    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", None)
    monkeypatch.setattr(dm.urllib.request, "urlopen", _fail_urlopen)

    out_pt = tmp_path / "missing_sha.pt"
    rc = dm.main(["--url", "https://example.com/model.pt", "--output", str(out_pt)])
    assert rc != 0
    assert called["urlopen"] == 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    err_txt = capsys.readouterr().err
    assert "SHA-256" in err_txt


def test_r4_download_models_wrong_sha_removes_part_and_no_target(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    payload = b"FAKE_WEIGHT_BYTES_FOR_SHA_CHECK"

    class FakeResp:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def read(self, n: int = -1):
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos:] if n < 0 else self._data[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))
    out_pt = tmp_path / "wrong_sha.pt"
    rc = dm.main([
        "--url", "https://example.com/model.pt",
        "--output", str(out_pt),
        "--sha256", "0" * 64,
    ])
    assert rc != 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    err_txt = capsys.readouterr().err
    assert "SHA-256" in err_txt


def test_r4_download_models_insecure_skip_sha256_warns_and_saves(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    payload = b"BYPASS_VERIFIED_BYTES"

    class FakeResp:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def read(self, n: int = -1):
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos:] if n < 0 else self._data[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", None)
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))

    out_pt = tmp_path / "insecure_ok.pt"
    rc = dm.main([
        "--url", "https://example.com/model.pt",
        "--output", str(out_pt),
        "--insecure-skip-sha256",
    ])
    assert rc == 0
    assert out_pt.is_file() and out_pt.read_bytes() == payload
    out_txt = capsys.readouterr().out
    assert "UYARI" in out_txt
    assert "SHA-256" in out_txt


# ----------------------------------------------------------------------
# R5 KABUL TESTİ: Geçersiz Video Yolunda Model Yüklenmeden Erken Hata
# ----------------------------------------------------------------------
def test_r5_invalid_input_exits_before_loading_object_detector(tmp_path, monkeypatch, capsys):
    import main as main_mod

    missing_video = tmp_path / "does_not_exist.mp4"
    export_dir = tmp_path / "should_not_create"
    detector_calls = {"count": 0}

    def _forbidden_detector(*args, **kwargs):
        detector_calls["count"] += 1
        raise AssertionError("Geçersiz video yolunda ObjectDetector başlatılmamalıydı!")

    monkeypatch.setattr(main_mod, "ObjectDetector", _forbidden_detector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(missing_video),
            "--export-dir",
            str(export_dir),
            "--no-video",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code != 0
    assert detector_calls["count"] == 0
    assert not export_dir.exists()
    err_txt = capsys.readouterr().err
    assert "bulunamadı" in err_txt
