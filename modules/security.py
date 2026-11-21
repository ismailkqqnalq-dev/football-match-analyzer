"""
security.py
-----------
Üretim ortamı ve web API yayını için merkezi güvenlik kalkanı modülü.

Stadyum güvenliği benzetmesiyle 5 temel riski engeller:
    1) Path Traversal & Symlink Koruması (`resolve_safe_path`):
       İzin verilen stadyum alanı (dizin) dışına `../` veya sembolik bağla
       sızma girişimlerini anında durdurur.
    2) Sahte Video / Zararlı Yükleme Koruması (`validate_video_file`):
       Sadece dosya uzantısına (.mp4) güvenmez; dosya başındaki gerçek ikili
       imzaya (magic bytes), çözünürlüğe (<= 4K), FPS'ye, süreye ve boyuta bakar.
    3) Model Pickle / RCE Koruması (`validate_model_weight_path`):
       PyTorch `.pt` dosyaları Python `pickle` çalıştırabildiği için dışarıdan
       gelen belirsiz yolları reddeder; yalnızca güvenli `models/` dizini ve
       opsiyonel SHA-256 özetiyle yükleme yapılmasına izin verir.
    4) SQL Tanımlayıcı & CSV Formül Enjeksiyonu Koruması:
       Tablo adlarını sıkı beyaz listeye alır; Excel'de kod çalıştırabilen
       `=`, `+`, `-`, `@` başlangıçlı hücreleri zararsızlaştırır.
    5) Kimlik Doğrulama (HMAC-SHA256 / JWT) ve Hız Sınırlandırıcı (`TokenBucketRateLimiter`):
       Biletsiz (yetkisiz) girişi ve GPU'yu kilitleyen ardışık yükleme (DDoS)
       saldırılarını durdurur.
"""

import hashlib
import hmac
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import cv2
import jwt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_WINDOWS_RESERVED_DRIVE_RE = re.compile(r"^[A-Za-z]:$")


class SecurityViolationError(ValueError):
    """Güvenlik politikası ihlal edildiğinde fırlatılan özel hata."""


# ----------------------------------------------------------------------
# 1. PATH TRAVERSAL & SYMLINK KORUMASI
# ----------------------------------------------------------------------
def resolve_safe_path(
    target_path: Union[str, os.PathLike],
    allowed_root: Union[str, os.PathLike],
    must_exist: bool = False,
) -> Path:
    """
    Verilen yolu izin verilen kök dizin (`allowed_root`) içine hapseder.
    Null byte, UNC ağ yolları, `..` tırmanışları ve dışarı kaçan sembolik
    bağlar `SecurityViolationError` ile reddedilir.
    """
    raw = os.fspath(target_path)
    if not isinstance(raw, str) or not raw.strip():
        raise SecurityViolationError("Geçersiz veya boş dosya yolu.")
    if "\x00" in raw:
        raise SecurityViolationError("Dosya yolunda yasaklanmış null karakter (\\x00) tespit edildi.")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise SecurityViolationError("Ağ (UNC) yollarına izin verilmiyor.")

    root_resolved = Path(allowed_root).resolve()
    p = Path(raw)

    # Göreceli yollar her zaman allowed_root altında değerlendirilir
    candidate = p if p.is_absolute() else (root_resolved / p)

    # Sembolik bağ bileşenlerinin izinli kök dışına işaret etmediğinden emin ol
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise SecurityViolationError(f"Dosya yolu güvenli şekilde çözümlenemedi: {exc}") from exc

    try:
        is_inside = resolved.is_relative_to(root_resolved)
    except ValueError:
        is_inside = False

    if not is_inside:
        raise SecurityViolationError(
            "Dizin dışına erişim girişimi engellendi (Path Traversal / Symlink)."
        )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {resolved}")

    return resolved


# ----------------------------------------------------------------------
# 2. BÜTÜNLÜK, SQL TANIMLAYICI & CSV FORMÜL TEMİZLEME
# ----------------------------------------------------------------------
def compute_sha256(file_path: Union[str, os.PathLike]) -> str:
    """Dosyanın SHA-256 özetini bellek dostu 64 KiB bloklarla hesaplar."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sql_identifier(name: str) -> str:
    """SQLite tablo/sütun adını katı alfanümerik desene (`[A-Za-z_][A-Za-z0-9_]*`) zorlar."""
    if not isinstance(name, str) or not _SQL_IDENTIFIER_RE.fullmatch(name):
        raise SecurityViolationError(
            f"Güvensiz SQL tanımlayıcısı reddedildi: {name!r}"
        )
    return name


def sanitize_csv_value(value: Any) -> Any:
    """
    CSV / Excel formül enjeksiyonunu (Formula Injection) önler.
    Metin değeri `=`, `+`, `-`, `@`, `\\t` veya `\\r` ile başlıyorsa başına
    tek tırnak (`'`) ekleyerek elektronik tablo programlarının komut çalıştırmasını engeller.
    """
    if isinstance(value, str) and value and value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


# ----------------------------------------------------------------------
# 3. VİDEO DOSYASI DERİN DENETİMİ (MAGIC BYTES & BOMB GUARD)
# ----------------------------------------------------------------------
def has_valid_video_magic_bytes(header: bytes) -> bool:
    """
    Dosya başındaki ilk baytları (magic bytes / container imzası) inceler.
    Uzantısı `.mp4` yapılmış çalıştırılabilir dosya, betik, HTML veya ZIP'leri eler.
    """
    if not header or len(header) < 12:
        return False

    # ISO Base Media / MP4 / MOV / M4V: ilk 4 byte box uzunluğu, [4:8] == b"ftyp"
    if header[4:8] == b"ftyp":
        return True

    # AVI konteyneri: RIFF....AVI
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return True

    # Matroska / WebM (MKV): EBML başlığı \x1a\x45\xdf\xa3
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return True

    return False


def validate_video_file(
    video_path: Union[str, os.PathLike],
    max_size_mb: Optional[float] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_duration_sec: Optional[float] = None,
    allowed_extensions: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Yüklenen veya işlenecek video dosyasını güvenlik ve kaynak sınırlarına göre denetler:
      - Uzantı beyaz listesi
      - Sıfır byte veya maksimum dosya boyutu kontrolü
      - İkili container imzası (magic bytes) kontrolü
      - OpenCV ile çözünürlük (decompression bomb koruması), FPS ve süre sınırları
      - En az 1 gerçek karenin çözülebilir olması
    """
    raw = os.fspath(video_path)
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise SecurityViolationError("Geçersiz video dosya yolu.")

    p = Path(raw)
    if not p.is_file():
        raise FileNotFoundError(f"Video dosyası bulunamadı: {p}")

    ext = p.suffix.lower()
    allowed_exts = (
        {e.lower() for e in allowed_extensions}
        if allowed_extensions is not None
        else {e.lower() for e in config.SECURITY_ALLOWED_VIDEO_EXTENSIONS}
    )
    if ext not in allowed_exts:
        raise SecurityViolationError(
            f"Desteklenmeyen veya izin verilmeyen video uzantısı: '{ext}'."
        )

    size_bytes = p.stat().st_size
    if size_bytes <= 0:
        raise SecurityViolationError("Boş (0 bayt) video dosyası kabul edilmiyor.")

    limit_mb = float(
        config.SECURITY_MAX_VIDEO_SIZE_MB if max_size_mb is None else max_size_mb
    )
    max_bytes = int(limit_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        raise SecurityViolationError(
            f"Video boyutu sınırı aşıldı ({size_bytes / (1024 * 1024):.1f} MB > {limit_mb:.1f} MB)."
        )

    with open(p, "rb") as f:
        header = f.read(32)
    if not has_valid_video_magic_bytes(header):
        raise SecurityViolationError(
            "Dosya içeriği geçerli bir video konteyner başlığı (MP4/MOV/AVI/MKV) taşımıyor."
        )

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise SecurityViolationError("Video akışı OpenCV tarafından açılamadı.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        w_limit = int(config.SECURITY_MAX_VIDEO_WIDTH if max_width is None else max_width)
        h_limit = int(config.SECURITY_MAX_VIDEO_HEIGHT if max_height is None else max_height)
        if width <= 0 or height <= 0:
            raise SecurityViolationError(
                f"Geçersiz video kare boyutları tespit edildi: {width}x{height}."
            )
        if width > w_limit or height > h_limit:
            raise SecurityViolationError(
                f"Video çözünürlüğü güvenlik sınırını aşıyor: {width}x{height} > {w_limit}x{h_limit}."
            )

        max_fps = float(config.SECURITY_MAX_VIDEO_FPS)
        min_fps = float(config.SECURITY_MIN_VIDEO_FPS)
        if not math.isfinite(fps) or fps < min_fps or fps > max_fps:
            raise SecurityViolationError(
                f"Geçersiz veya sınır dışı video FPS değeri: {fps:.2f}."
            )

        dur_limit = float(
            config.SECURITY_MAX_VIDEO_DURATION_SEC
            if max_duration_sec is None
            else max_duration_sec
        )
        duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0
        if duration_sec > dur_limit:
            raise SecurityViolationError(
                f"Video süresi maksimum sınırı aşıyor ({duration_sec:.1f}s > {dur_limit:.1f}s)."
            )

        ok, first_frame = cap.read()
        if not ok or first_frame is None or first_frame.size == 0:
            raise SecurityViolationError("Videodan ilk kare çözümlenemedi (bozuk akış).")

        return {
            "size_bytes": int(size_bytes),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 2),
        }
    finally:
        cap.release()


# ----------------------------------------------------------------------
# 4. MODEL AĞIRLIĞI (PICKLE / RCE) VE SHA-256 DOĞRULAMASI
# ----------------------------------------------------------------------
def validate_model_weight_path(
    model_path: Union[str, os.PathLike],
    enforce_whitelist: Optional[bool] = None,
    expected_sha256: Optional[str] = None,
    models_dir: Optional[Union[str, os.PathLike]] = None,
) -> Path:
    """
    PyTorch `.pt` (pickle) deserialization / RCE riskine karşı model yolunu doğrular:
      - Null byte ve yol tırmanışı (`..`) engellenir.
      - Resmi Ultralytics taban isimleri (`yolo11s.pt`, `yolov8x.pt` vb.) yalnız
        yalın dosya adı olarak verilirse kabul edilir.
      - Yerel `.pt` ağırlıkları yalnızca `config.MODELS_DIR` (veya `models_dir`)
        içinden ve sembolik bağla dışarı taşmayacak biçimde yüklenebilir.
      - `expected_sha256` (veya `config.SECURITY_MODEL_SHA256`) tanımlıysa dosyanın
        SHA-256 özeti sabit zamanlı karşılaştırmayla doğrulanır.
    """
    raw = os.fspath(model_path)
    if not isinstance(raw, str) or not raw.strip():
        raise SecurityViolationError("Model ağırlık yolu boş olamaz.")
    if "\x00" in raw:
        raise SecurityViolationError("Model yolunda yasaklanmış null karakter tespit edildi.")

    p = Path(raw)
    if ".." in p.parts:
        raise SecurityViolationError(f"Model yolunda '..' kullanımına izin verilmiyor: {raw}")

    should_enforce = (
        bool(getattr(config, "SECURITY_ENFORCE_MODEL_DIR_WHITELIST", True))
        if enforce_whitelist is None
        else bool(enforce_whitelist)
    )
    official_basenames = set(
        getattr(config, "SECURITY_ALLOWED_OFFICIAL_YOLO_BASENAMES", set())
    )
    is_bare_name = (raw == p.name) and ("/" not in raw) and ("\\" not in raw)
    allowed_models_root = Path(models_dir if models_dir is not None else config.MODELS_DIR).resolve()

    if should_enforce:
        if p.suffix.lower() != ".pt":
            raise SecurityViolationError(
                f"Yalnızca '.pt' model uzantısına izin verilir: '{p.suffix}'."
            )

        if is_bare_name and raw in official_basenames:
            # Resmi Ultralytics temel ağırlık ismi (ör. yolo11s.pt)
            resolved_model = Path(raw)
            if (allowed_models_root / raw).is_file():
                resolved_model = (allowed_models_root / raw).resolve()
        else:
            # Yerel ağırlık dosyası mutlaka models/ dizini içinde olmalı
            if not p.is_absolute():
                candidate = (allowed_models_root / p) if is_bare_name else Path(raw)
            else:
                candidate = p
            resolved_model = resolve_safe_path(candidate, allowed_models_root, must_exist=False)
    else:
        resolved_model = p.resolve()

    sha_target = (
        expected_sha256
        if expected_sha256 is not None
        else getattr(config, "SECURITY_MODEL_SHA256", None)
    )
    if sha_target and resolved_model.is_file():
        actual_sha = compute_sha256(resolved_model)
        if not hmac.compare_digest(actual_sha.lower(), str(sha_target).strip().lower()):
            raise SecurityViolationError(
                f"Model SHA-256 bütünlük doğrulaması başarısız oldu: {resolved_model.name}"
            )

    return resolved_model


# ----------------------------------------------------------------------
# 5. KİMLİK DOĞRULAMA (API KEY / JWT) & HIZ SINIRLANDIRICI
# ----------------------------------------------------------------------
def verify_api_key(provided_key: Optional[str], expected_key: Optional[str]) -> bool:
    """Zamanlama saldırılarına (timing attack) karşı sabit zamanlı API anahtarı kontrolü."""
    if not provided_key or not expected_key:
        return False
    prov_b = str(provided_key).encode("utf-8")
    exp_b = str(expected_key).encode("utf-8")
    return hmac.compare_digest(prov_b, exp_b)


def create_access_token(
    subject: str,
    secret: str,
    ttl_seconds: int = 3600,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Kısa ömürlü, HS256 imzalı erişim (JWT) bileti üretir."""
    if not secret or len(str(secret)) < 16:
        raise ValueError("JWT imzalama gizli anahtarı en az 16 karakter olmalıdır.")
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + max(1, int(ttl_seconds)),
    }
    if extra_claims:
        for k, v in extra_claims.items():
            if k not in {"sub", "iat", "exp"}:
                payload[k] = v
    return str(jwt.encode(payload, secret, algorithm="HS256"))


def verify_access_token(token: str, secret: str) -> Dict[str, Any]:
    """Gelen JWT biletini imza, son kullanma tarihi (`exp`) ve konu (`sub`) açısından doğrular."""
    if not token or not secret:
        raise SecurityViolationError("Eksik kimlik doğrulama bileti.")
    try:
        decoded = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp", "sub"]},
        )
        return dict(decoded)
    except jwt.ExpiredSignatureError as exc:
        raise SecurityViolationError("Erişim biletinin (JWT) süresi dolmuş.") from exc
    except jwt.InvalidTokenError as exc:
        raise SecurityViolationError("Geçersiz erişim bileti (JWT).") from exc


class TokenBucketRateLimiter:
    """
    İstemci başına istek sayısını dakikada sınırlayan, iş parçacığı güvenli
    (thread-safe) ve bellek taşmasına karşı LRU kapasiteli hız kesici.
    """

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: float = 60.0,
        max_tracked_keys: int = 10_000,
    ):
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_tracked_keys = max(100, int(max_tracked_keys))
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()

    def allow(self, key: str) -> Tuple[bool, float]:
        """
        İstemci anahtarı için 1 jeton düşmeye çalışır.
        Dönüş: (izin_verildi_mi, tekrar_deneme_bekleme_saniyesi)
        """
        now = time.monotonic()
        rate = float(self.max_requests) / self.window_seconds

        with self._lock:
            entry = self._buckets.get(key)
            if entry is None:
                tokens = float(self.max_requests)
                last_ts = now
            else:
                tokens, last_ts = entry
                elapsed = max(0.0, now - last_ts)
                tokens = min(float(self.max_requests), tokens + elapsed * rate)
                last_ts = now

            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = (tokens, last_ts)
                self._buckets.move_to_end(key)
                if len(self._buckets) > self.max_tracked_keys:
                    self._buckets.popitem(last=False)
                return True, 0.0

            retry_after = max(0.01, (1.0 - tokens) / max(rate, 1e-9))
            self._buckets[key] = (tokens, last_ts)
            self._buckets.move_to_end(key)
            return False, retry_after
