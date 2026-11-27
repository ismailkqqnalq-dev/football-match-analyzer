"""
utils/download_models.py
------------------------
Futbol maç analiz pipeline'ı için eğitilmiş YOLO nesne tespit ağırlığını
(`models/football_yolo11s_best.pt`) veya 32 noktalı saha kalibrasyon YOLO-Pose
ağırlığını (`models/football_pitch_pose.pt`) uzak bir URL'den indirir ve
SHA-256 bütünlüğünü doğrular.

Kullanım:
    python utils/download_models.py --target detector
    python utils/download_models.py --target pitch --url https://.../football_pitch_pose.pt --sha256 <SHA256>
"""

import argparse
import hashlib
import hmac
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import config  # noqa: E402

_TARGET_SPECS: Dict[str, Dict[str, str]] = {
    "detector": {
        "label": "4 sınıflı futbol YOLO nesne tespit modeli",
        "url_attr": "MODEL_DOWNLOAD_URL",
        "path_attr": "YOLO_MODEL_PATH",
        "sha_attr": "SECURITY_MODEL_SHA256",
        "env_var": "FOOTBALL_MODEL_SHA256",
        "train_cmd": "python utils/train_custom_yolo.py",
    },
    "pitch": {
        "label": "32 noktalı saha kalibrasyon YOLO-Pose modeli",
        "url_attr": "PITCH_KEYPOINT_MODEL_DOWNLOAD_URL",
        "path_attr": "PITCH_KEYPOINT_MODEL_PATH",
        "sha_attr": "SECURITY_PITCH_MODEL_SHA256",
        "env_var": "FOOTBALL_PITCH_MODEL_SHA256",
        "train_cmd": "python utils/train_pitch_keypoints.py",
    },
}


def _resolve_target_spec(target: str) -> Dict[str, str]:
    """İndirme hedefine karşılık gelen config ve ortam değişkeni eşlemesini döndürür."""
    key = (target or "detector").strip().lower()
    if key not in _TARGET_SPECS:
        raise ValueError(
            f"Geçersiz model hedefi: {target!r}. Geçerli değerler: {', '.join(_TARGET_SPECS)}"
        )
    return _TARGET_SPECS[key]


def compute_sha256(file_path: Path) -> str:
    """Dosyanın SHA-256 özetini parçalı okuma ile hesaplar."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    url: Optional[str] = None,
    output_path: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
    overwrite: bool = False,
    timeout_sec: float = 120.0,
    insecure_skip_sha256: bool = False,
    target: str = "detector",
) -> Path:
    """
    Seçilen hedef (`detector` veya `pitch`) için model ağırlığını belirtilen
    URL'den indirir ve SHA-256 doğrulamasından geçirdikten sonra hedef konuma
    atomik olarak taşır. SHA-256 zorunludur; bilinçli atlama için
    `insecure_skip_sha256=True` gerekir.
    """
    target_key = (target or "detector").strip().lower()
    spec = _resolve_target_spec(target_key)
    url_attr = spec["url_attr"]
    path_attr = spec["path_attr"]
    sha_attr = spec["sha_attr"]
    env_var = spec["env_var"]
    train_cmd = spec["train_cmd"]

    default_output = getattr(config, path_attr)
    target_path = Path(output_path if output_path else default_output).resolve()
    sha_expected = (
        expected_sha256
        if expected_sha256 is not None
        else (getattr(config, sha_attr, None) or "")
    ).strip().lower()

    if target_path.is_file() and not overwrite:
        if sha_expected:
            actual = compute_sha256(target_path).lower()
            if not hmac.compare_digest(actual, sha_expected):
                raise ValueError(
                    f"Mevcut model dosyası SHA-256 eşleşmedi ({actual} != {sha_expected}). "
                    "Yeniden indirmek için --overwrite bayrağını kullanın."
                )
        print(f"[download_models] Model ağırlığı zaten mevcut: {target_path}")
        return target_path

    resolved_url = (url if url is not None else getattr(config, url_attr, "")).strip()
    if not resolved_url:
        raise ValueError(
            f"Model indirme adresi tanımlı değil (`config.{url_attr}` boş).\n"
            "Çözüm seçenekleri:\n"
            f"  1) İndirme bağlantısını `--target {target_key} --url <URL>` ile verin veya "
            f"`config/config.py` içinde `{url_attr}` alanını doldurun.\n"
            f"  2) Modeli kendi GPU ortamınızda sıfırdan eğitmek için `{train_cmd}` "
            "komutunu çalıştırın.\n"
            "  Ayrıntılar için README.md içindeki 'Model ağırlıkları' bölümüne bakın."
        )

    parsed = urllib.parse.urlparse(resolved_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Geçersiz indirme URL'si (yalnızca http/https desteklenir): {resolved_url}")

    if not sha_expected:
        if not insecure_skip_sha256:
            raise ValueError(
                "SHA-256 sağlama toplamı tanımlı değil (`--sha256` verilmedi ve "
                f"`config.{sha_attr}` / `{env_var}` boş).\n"
                "PyTorch .pt dosyaları yüklenirken keyfi kod çalıştırabildiğinden doğrulanmamış "
                "indirme güvenlik riski taşır ve varsayılan olarak reddedilir.\n"
                "Doğrulamayı bilerek atlamak istiyorsanız `--insecure-skip-sha256` bayrağını ekleyin."
            )
        print(
            "[download_models] UYARI: --insecure-skip-sha256 ETKİN – SHA-256 DOĞRULAMASI ATLANDI! "
            "DOĞRULANMAMIŞ .pt DOSYASI GÜVENLİK RİSKİ TAŞIR."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    print(f"[download_models] {spec['label']} indiriliyor: {resolved_url} -> {target_path}")
    try:
        req = urllib.request.Request(resolved_url, headers={"User-Agent": "football-match-analyzer/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp_path, "wb") as fout:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)

        actual_sha = compute_sha256(tmp_path).lower()
        if sha_expected and not hmac.compare_digest(actual_sha, sha_expected):
            raise ValueError(
                f"İndirilen model SHA-256 doğrulamasından geçemedi!\n"
                f"  Beklenen : {sha_expected}\n"
                f"  Hesaplanan: {actual_sha}"
            )

        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    print(f"[download_models] Model başarıyla kaydedildi: {target_path} (SHA-256: {actual_sha})")
    return target_path


def parse_args(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Eğitilmiş futbol YOLO nesne tespit veya 32 noktalı saha kalibrasyon (YOLO-Pose) "
            "model ağırlığını indirir ve SHA-256 doğrular."
        )
    )
    parser.add_argument(
        "--target",
        "--model",
        dest="target",
        choices=list(_TARGET_SPECS.keys()),
        default="detector",
        help="İndirilecek model hedefi: 'detector' (varsayılan) veya 'pitch' (saha anahtar nokta modeli)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=(
            "Model ağırlığı indirme adresi "
            "(varsayılan: hedefe göre config.MODEL_DOWNLOAD_URL veya config.PITCH_KEYPOINT_MODEL_DOWNLOAD_URL)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Kaydedilecek hedef .pt dosya yolu "
            "(varsayılan: hedefe göre config.YOLO_MODEL_PATH veya config.PITCH_KEYPOINT_MODEL_PATH)"
        ),
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help=(
            "Beklenen SHA-256 özeti "
            "(varsayılan: hedefe göre config.SECURITY_MODEL_SHA256 / FOOTBALL_MODEL_SHA256 "
            "veya config.SECURITY_PITCH_MODEL_SHA256 / FOOTBALL_PITCH_MODEL_SHA256)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hedef dosya varsa üzerine yaz",
    )
    parser.add_argument(
        "--insecure-skip-sha256",
        action="store_true",
        help="UYARI: SHA-256 doğrulamasını bilinçli olarak atla (yalnızca güvenilir kaynaklar için)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        download_model(
            url=args.url,
            output_path=Path(args.output) if args.output is not None else None,
            expected_sha256=args.sha256,
            overwrite=bool(args.overwrite),
            insecure_skip_sha256=bool(args.insecure_skip_sha256),
            target=args.target,
        )
        return 0
    except (ValueError, urllib.error.URLError, OSError) as exc:
        print(f"[download_models] Hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
