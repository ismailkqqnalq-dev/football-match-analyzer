"""
api/app.py
----------
Futbol Maç Analiz Sistemi için Güvenli Üretim (Production) REST API ve
Sıralı GPU İş Kuyruğu katmanı.

Güvenlik Katmanları:
    1) Kimlik Doğrulama (JWT Bearer / sabit zamanlı X-API-Key)
    2) Bellek Sınırlandırmalı Hız Kesici (TokenBucketRateLimiter - Genel & Upload)
    3) HTTP Güvenlik Başlıkları ve Katı CORS Politikası
    4) Akışlı (Chunked) Boyut Sınırlı Yükleme + Deep Video X-Ray (Magic Bytes, <=4K, Süre)
    5) UUIDv4 İzole İş Klasörleri + Sınırlı GPU İş Kuyruğu (DDoS Koruması)
    6) Beyaz Listeli & Path-Traversal Korumalı Artefakt İndirme
"""

from contextlib import asynccontextmanager
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import config
from modules.security import (
    SecurityViolationError,
    TokenBucketRateLimiter,
    create_access_token,
    resolve_safe_path,
    validate_video_file,
    verify_access_token,
    verify_api_key,
)

_MAIN_PY = _REPO_ROOT / "main.py"
_ALLOWED_TRACKERS = {"deepsort", "bytetrack"}
_SAFE_PNG_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.png$")
_ALLOWED_TOP_LEVEL_ARTIFACTS = {
    "match_analytics_summary.json",
    "possession_timeline.csv",
    "player_distance_summary.csv",
    "tracking_data.csv",
    "tracking_data.db",
}
_ALLOWED_ML_ARTIFACTS = {
    "ml_analysis_summary.json",
    "ml_analysis_report.md",
    "ml_player_profiles.csv",
    "ml_tempo_windows.csv",
    "ml_anomalies.csv",
    "ml_pass_predictions.csv",
}
_STOP_SENTINEL = object()


def _is_allowed_artifact_name(artifact_name: str) -> bool:
    """Yalnızca güvenli, beyaz listedeki göreceli rapor/çıktı yollarını kabul eder."""
    if not artifact_name or "\x00" in artifact_name or "\\" in artifact_name:
        return False
    if artifact_name.startswith("/"):
        return False
    parts = Path(artifact_name).parts
    if any(part in {"", ".", ".."} for part in parts):
        return False

    if len(parts) == 1:
        return parts[0] in _ALLOWED_TOP_LEVEL_ARTIFACTS

    if len(parts) == 2:
        folder, fname = parts
        if folder == "ml_reports":
            return (fname in _ALLOWED_ML_ARTIFACTS) or bool(_SAFE_PNG_NAME_RE.fullmatch(fname))
        if folder == "heatmaps":
            return bool(_SAFE_PNG_NAME_RE.fullmatch(fname))

    return False


def _collect_job_artifacts(export_dir: Path) -> List[str]:
    """İzin verilen çıktı dosyalarını göreceli yol listesi olarak toplar."""
    found: List[str] = []
    if not export_dir.is_dir():
        return found
    for name in sorted(_ALLOWED_TOP_LEVEL_ARTIFACTS):
        if (export_dir / name).is_file():
            found.append(name)
    for sub in ("ml_reports", "heatmaps"):
        sub_dir = export_dir / sub
        if sub_dir.is_dir():
            for child in sorted(sub_dir.iterdir()):
                rel = f"{sub}/{child.name}"
                if child.is_file() and _is_allowed_artifact_name(rel):
                    found.append(rel)
    return found


def _default_pipeline_executor(
    input_path: Path,
    export_dir: Path,
    tracker: str = "deepsort",
    flush_interval: int = 2500,
) -> None:
    """İzole alt süreç (`shell=False`) üzerinden analiz pipeline'ını çalıştırır."""
    safe_env = os.environ.copy()
    safe_env.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "ultralytics_cfg"),
    )
    cmd = [
        sys.executable,
        str(_MAIN_PY),
        "--input",
        str(input_path),
        "--no-video",
        "--export-dir",
        str(export_dir),
        "--flush-interval",
        str(int(flush_interval)),
        "--tracker",
        tracker if tracker in _ALLOWED_TRACKERS else "deepsort",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=safe_env,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Analiz motoru hata koduyla sonlandı (returncode={res.returncode}).")


class BoundedJobQueue:
    """
    GPU kaynak tükenmesini (DDoS) önlemek için analiz işlerini sınırlı kapasiteli
    bir kuyrukta sıraya sokan ve eşzamanlı işçi sayısını kısıtlayan yönetici.
    """

    def __init__(
        self,
        jobs_dir: Path,
        max_queued_jobs: int = 16,
        max_workers: int = 1,
        executor: Optional[Callable[[Path, Path, str, int], None]] = None,
    ):
        self.jobs_dir = Path(jobs_dir).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_queued_jobs = max(1, int(max_queued_jobs))
        self.max_workers = max(1, int(max_workers))
        self._executor = executor or _default_pipeline_executor

        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=self.max_queued_jobs)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._started = False
        self._stopped = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stopped = False
            self._workers.clear()
            for idx in range(self.max_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"football-gpu-worker-{idx}",
                    daemon=True,
                )
                t.start()
                self._workers.append(t)
            self._started = True

    def shutdown(self, timeout_sec: float = 2.0) -> None:
        with self._lock:
            if not self._started or self._stopped:
                return
            self._stopped = True
            workers = list(self._workers)

        for _ in workers:
            try:
                self._queue.put_nowait(_STOP_SENTINEL)
            except queue.Full:
                break

        for t in workers:
            t.join(timeout=timeout_sec)

        with self._lock:
            self._started = False
            self._workers.clear()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP_SENTINEL:
                self._queue.task_done()
                break

            job_id, input_path, export_dir, tracker, flush_interval = item
            try:
                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is None:
                        continue
                    rec["status"] = "running"
                    rec["started_at"] = time.time()

                self._executor(input_path, export_dir, tracker, flush_interval)
                artifacts = _collect_job_artifacts(export_dir)

                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is not None:
                        rec["status"] = "completed"
                        rec["completed_at"] = time.time()
                        rec["artifacts"] = artifacts
            except Exception as exc:
                safe_type = type(exc).__name__
                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is not None:
                        rec["status"] = "failed"
                        rec["completed_at"] = time.time()
                        rec["error"] = f"Analiz sırasında işlem hatası oluştu ({safe_type})."
            finally:
                self._queue.task_done()

    def enqueue(
        self,
        job_id: str,
        input_path: Path,
        export_dir: Path,
        video_meta: Dict[str, Any],
        tracker: str = "deepsort",
        flush_interval: int = 2500,
    ) -> Dict[str, Any]:
        if not self._started:
            self.start()

        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "video_meta": video_meta,
            "artifacts": [],
            "_export_dir": export_dir,
        }
        with self._lock:
            self._jobs[job_id] = record

        try:
            self._queue.put_nowait((job_id, input_path, export_dir, tracker, flush_interval))
        except queue.Full as exc:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise RuntimeError("GPU analiz kuyruğu dolu; lütfen daha sonra tekrar deneyin.") from exc

        return self.get_public_record(job_id) or {}

    def get_public_record(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return {
                "job_id": rec["job_id"],
                "status": rec["status"],
                "created_at": rec["created_at"],
                "started_at": rec["started_at"],
                "completed_at": rec["completed_at"],
                "error": rec["error"],
                "video_meta": rec["video_meta"],
                "artifacts": list(rec["artifacts"]),
            }

    def get_export_dir(self, job_id: str) -> Optional[Path]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return Path(rec["_export_dir"])

    @property
    def queue_depth(self) -> int:
        return int(self._queue.qsize())


def _validate_uuid4(raw_id: str) -> str:
    try:
        parsed = uuid.UUID(str(raw_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Geçersiz iş kimliği (UUID formatı bekleniyor).",
        ) from exc
    if parsed.version != 4:
        raise HTTPException(
            status_code=400,
            detail="Geçersiz iş kimliği (UUIDv4 bekleniyor).",
        )
    return str(parsed)


def create_app(
    jobs_dir: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    jwt_ttl_seconds: Optional[int] = None,
    max_upload_mb: Optional[float] = None,
    max_video_width: Optional[int] = None,
    max_video_height: Optional[int] = None,
    max_video_duration_sec: Optional[float] = None,
    rate_limit_per_min: Optional[int] = None,
    upload_rate_limit_per_min: Optional[int] = None,
    max_queued_jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    job_executor: Optional[Callable[[Path, Path, str, int], None]] = None,
) -> FastAPI:
    """Güvenlik sertleştirmeli FastAPI uygulamasını yapılandırır ve döndürür."""
    resolved_jobs_dir = Path(jobs_dir if jobs_dir is not None else config.API_JOBS_DIR).resolve()
    configured_api_key = api_key if api_key is not None else (os.environ.get("FOOTBALL_API_KEY") or None)
    configured_jwt_secret = (
        jwt_secret
        if jwt_secret is not None
        else (os.environ.get("FOOTBALL_JWT_SECRET") or secrets.token_urlsafe(32))
    )
    resolved_ttl = int(jwt_ttl_seconds if jwt_ttl_seconds is not None else config.API_JWT_TTL_SECONDS)
    upload_limit_mb = float(
        max_upload_mb if max_upload_mb is not None else config.SECURITY_MAX_VIDEO_SIZE_MB
    )
    max_upload_bytes = max(1, int(upload_limit_mb * 1024 * 1024))

    general_limiter = TokenBucketRateLimiter(
        max_requests=int(
            rate_limit_per_min
            if rate_limit_per_min is not None
            else config.API_RATE_LIMIT_REQUESTS_PER_MIN
        ),
        window_seconds=60.0,
    )
    upload_limiter = TokenBucketRateLimiter(
        max_requests=int(
            upload_rate_limit_per_min
            if upload_rate_limit_per_min is not None
            else config.API_UPLOAD_RATE_LIMIT_PER_MIN
        ),
        window_seconds=60.0,
    )
    job_queue = BoundedJobQueue(
        jobs_dir=resolved_jobs_dir,
        max_queued_jobs=int(
            max_queued_jobs if max_queued_jobs is not None else config.API_MAX_QUEUED_JOBS
        ),
        max_workers=int(
            max_workers if max_workers is not None else config.API_MAX_CONCURRENT_GPU_JOBS
        ),
        executor=job_executor,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        job_queue.start()
        try:
            yield
        finally:
            job_queue.shutdown()

    api_app = FastAPI(
        title="Futbol Maç Analiz API",
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    api_app.state.job_queue = job_queue

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.API_ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )

    @api_app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    def _client_key(request: Request) -> str:
        return request.client.host if (request.client and request.client.host) else "unknown"

    def _enforce_general_rate_limit(request: Request) -> None:
        allowed, retry_after = general_limiter.allow(f"gen:{_client_key(request)}")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Çok fazla istek gönderildi. Lütfen kısa süre sonra tekrar deneyin.",
                headers={"Retry-After": str(max(1, int(round(retry_after))))},
            )

    def _enforce_upload_rate_limit(request: Request) -> None:
        allowed, retry_after = upload_limiter.allow(f"upl:{_client_key(request)}")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Video yükleme hız sınırı aşıldı. Lütfen biraz bekleyin.",
                headers={"Retry-After": str(max(1, int(round(retry_after))))},
            )

    def _require_auth(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> str:
        if not configured_api_key:
            raise HTTPException(
                status_code=503,
                detail="Sunucu API kimlik doğrulama anahtarı yapılandırılmamış.",
            )

        if x_api_key is not None:
            if verify_api_key(x_api_key, configured_api_key):
                return "api_key_client"
            raise HTTPException(
                status_code=401,
                detail="Geçersiz API anahtarı.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            try:
                claims = verify_access_token(token, configured_jwt_secret)
                return str(claims["sub"])
            except SecurityViolationError as exc:
                raise HTTPException(
                    status_code=401,
                    detail=str(exc),
                    headers={"WWW-Authenticate": "Bearer"},
                ) from exc

        raise HTTPException(
            status_code=401,
            detail="Kimlik doğrulama gerekli (Bearer JWT veya X-API-Key).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @api_app.get("/api/v1/health")
    def health(_rl: None = Depends(_enforce_general_rate_limit)) -> Dict[str, Any]:
        return {
            "status": "ok",
            "queue_depth": job_queue.queue_depth,
            "max_queued_jobs": job_queue.max_queued_jobs,
            "auth_configured": bool(configured_api_key),
        }

    @api_app.post("/api/v1/auth/token")
    def issue_token(
        _rl: None = Depends(_enforce_general_rate_limit),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        if not configured_api_key:
            raise HTTPException(
                status_code=503,
                detail="Sunucu API kimlik doğrulama anahtarı yapılandırılmamış.",
            )
        if not verify_api_key(x_api_key, configured_api_key):
            raise HTTPException(
                status_code=401,
                detail="Geçersiz API anahtarı.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = create_access_token(
            subject="api_client",
            secret=configured_jwt_secret,
            ttl_seconds=resolved_ttl,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": resolved_ttl,
        }

    @api_app.post("/api/v1/jobs", status_code=202)
    async def submit_job(
        _rl: None = Depends(_enforce_general_rate_limit),
        _upl_rl: None = Depends(_enforce_upload_rate_limit),
        _sub: str = Depends(_require_auth),
        file: UploadFile = File(...),
        tracker: str = Form("deepsort"),
        flush_interval: int = Form(2500),
    ) -> Dict[str, Any]:
        norm_tracker = str(tracker).strip().lower()
        if norm_tracker not in _ALLOWED_TRACKERS:
            raise HTTPException(
                status_code=400,
                detail=f"Geçersiz tracker seçimi. İzin verilenler: {sorted(_ALLOWED_TRACKERS)}",
            )

        raw_ext = Path(file.filename or "upload.mp4").suffix.lower()
        if raw_ext not in config.SECURITY_ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"İzin verilmeyen video dosya uzantısı: '{raw_ext}'.",
            )

        job_id = str(uuid.uuid4())
        job_root = resolve_safe_path(job_id, resolved_jobs_dir)
        export_dir = resolve_safe_path("exports", job_root)
        input_path = resolve_safe_path(f"input{raw_ext}", job_root)

        job_root.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        total_written = 0
        chunk_size = 1024 * 1024
        try:
            with open(input_path, "wb") as f_out:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Video boyutu maksimum yükleme sınırını ({upload_limit_mb:.2f} MB) aşıyor.",
                        )
                    f_out.write(chunk)

            video_meta = validate_video_file(
                input_path,
                max_size_mb=upload_limit_mb,
                max_width=max_video_width,
                max_height=max_video_height,
                max_duration_sec=max_video_duration_sec,
            )

            return job_queue.enqueue(
                job_id=job_id,
                input_path=input_path,
                export_dir=export_dir,
                video_meta=video_meta,
                tracker=norm_tracker,
                flush_interval=max(0, int(flush_interval)),
            )
        except HTTPException:
            shutil.rmtree(job_root, ignore_errors=True)
            raise
        except SecurityViolationError as exc:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="Yüklenen video doğrulanamadı veya işleme alınamadı.",
            ) from exc

    @api_app.get("/api/v1/jobs/{job_id}")
    def get_job_status(
        job_id: str,
        _rl: None = Depends(_enforce_general_rate_limit),
        _sub: str = Depends(_require_auth),
    ) -> Dict[str, Any]:
        clean_id = _validate_uuid4(job_id)
        rec = job_queue.get_public_record(clean_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Analiz işi bulunamadı.")
        return rec

    @api_app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_name:path}")
    def download_artifact(
        job_id: str,
        artifact_name: str,
        _rl: None = Depends(_enforce_general_rate_limit),
        _sub: str = Depends(_require_auth),
    ):
        clean_id = _validate_uuid4(job_id)
        if not _is_allowed_artifact_name(artifact_name):
            raise HTTPException(
                status_code=403,
                detail="İstenen çıktı dosyası güvenlik beyaz listesinde değil.",
            )

        export_dir = job_queue.get_export_dir(clean_id)
        if export_dir is None:
            raise HTTPException(status_code=404, detail="Analiz işi bulunamadı.")

        try:
            target_file = resolve_safe_path(artifact_name, export_dir, must_exist=True)
        except SecurityViolationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="İstenen çıktı dosyası henüz üretilmemiş.") from exc

        return FileResponse(
            path=str(target_file),
            filename=target_file.name,
            content_disposition_type="attachment",
        )

    return api_app


app = create_app()
