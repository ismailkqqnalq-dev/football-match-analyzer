"""
utils/download_models.py
------------------------
Futbol maç analiz pipeline'ı için eğitilmiş YOLO ağırlık dosyasını
(`models/football_yolo11s_best.pt`) uzak bir URL'den indirir ve
isteğe bağlı olarak SHA-256 bütünlüğünü doğrular.

Kullanım:
    python utils/download_models.py
    python utils/download_models.py --url https://.../football_yolo11s_best.pt
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import config  # noqa: E402


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
) -> Path:
    """
    Model ağırlığını belirtilen URL'den indirir ve SHA-256 doğrulamasından
    geçirdikten sonra hedef konuma atomik olarak taşır. SHA-256 zorunludur;
    bilinçli atlama için `insecure_skip_sha256=True` gerekir.
    """
    resolved_url = (url if url is not None else getattr(config, "MODEL_DOWNLOAD_URL", "")).strip()
    if not resolved_url:
        raise ValueError(
            "Model indirme adresi tanımlı değil (`config.MODEL_DOWNLOAD_URL` boş).\n"
            "Çözüm seçenekleri:\n"
            "  1) İndirme bağlantısını `--url <URL>` ile verin veya `config/config.py` içinde "
            "`MODEL_DOWNLOAD_URL` alanını doldurun.\n"
            "  2) Modeli kendi GPU ortamınızda sıfırdan eğitmek için "
            "`python utils/train_custom_yolo.py` komutunu çalıştırın.\n"
            "  Ayrıntılar için README.md içindeki 'Model ağırlıkları' bölümüne bakın."
        )

    parsed = urllib.parse.urlparse(resolved_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Geçersiz indirme URL'si (yalnızca http/https desteklenir): {resolved_url}")

    target_path = Path(output_path or config.YOLO_MODEL_PATH).resolve()
    sha_expected = (
        expected_sha256
        if expected_sha256 is not None
        else (getattr(config, "SECURITY_MODEL_SHA256", None) or "")
    ).strip().lower()

    if not sha_expected:
        if not insecure_skip_sha256:
            raise ValueError(
                "SHA-256 sağlama toplamı tanımlı değil (`--sha256` verilmedi ve "
                "`config.SECURITY_MODEL_SHA256` / `FOOTBALL_MODEL_SHA256` boş).\n"
                "PyTorch .pt dosyaları yüklenirken keyfi kod çalıştırabildiğinden doğrulanmamış "
                "indirme güvenlik riski taşır ve varsayılan olarak reddedilir.\n"
                "Doğrulamayı bilerek atlamak istiyorsanız `--insecure-skip-sha256` bayrağını ekleyin."
            )
        print(
            "[download_models] UYARI: --insecure-skip-sha256 ETKİN – SHA-256 DOĞRULAMASI ATLANDI! "
            "DOĞRULANMAMIŞ .pt DOSYASI GÜVENLİK RİSKİ TAŞIR."
        )

    if target_path.is_file() and not overwrite:
        if sha_expected:
            actual = compute_sha256(target_path).lower()
            if actual != sha_expected:
                raise ValueError(
                    f"Mevcut model dosyası SHA-256 eşleşmedi ({actual} != {sha_expected}). "
                    "Yeniden indirmek için --overwrite bayrağını kullanın."
                )
        print(f"[download_models] Model ağırlığı zaten mevcut: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    print(f"[download_models] Model indiriliyor: {resolved_url} -> {target_path}")
    try:
        req = urllib.request.Request(resolved_url, headers={"User-Agent": "football-match-analyzer/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp_path, "wb") as fout:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    actual_sha = compute_sha256(tmp_path).lower()
    if sha_expected and actual_sha != sha_expected:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"İndirilen model SHA-256 doğrulamasından geçemedi!\n"
            f"  Beklenen : {sha_expected}\n"
            f"  Hesaplanan: {actual_sha}"
        )

    os.replace(tmp_path, target_path)
    print(f"[download_models] Model başarıyla kaydedildi: {target_path} (SHA-256: {actual_sha})")
    return target_path


def parse_args(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(
        description="Eğitilmiş 4 sınıflı futbol YOLO model ağırlığını indirir ve SHA-256 doğrular."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Model ağırlığı indirme adresi (varsayılan: config.MODEL_DOWNLOAD_URL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.YOLO_MODEL_PATH,
        help="Kaydedilecek hedef .pt dosya yolu (varsayılan: config.YOLO_MODEL_PATH)",
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Beklenen SHA-256 özeti (varsayılan: config.SECURITY_MODEL_SHA256 / FOOTBALL_MODEL_SHA256)",
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
            output_path=Path(args.output),
            expected_sha256=args.sha256,
            overwrite=bool(args.overwrite),
            insecure_skip_sha256=bool(args.insecure_skip_sha256),
        )
        return 0
    except (ValueError, urllib.error.URLError, OSError) as exc:
        print(f"[download_models] Hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
