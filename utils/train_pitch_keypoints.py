"""
train_pitch_keypoints.py
------------------------
SAHA ANAHTAR NOKTA (32 KEYPOINT) YOLO-POSE KALİBRASYON MODELİ EĞİTİM BETİĞİ

Bu betik, Roboflow üzerindeki `football-field-detection` (32 saha anahtar
noktası) veri setini geçici dizin (`%TEMP%`) altına indirip yerel NVIDIA GPU
(ör. RTX 4050 6 GB VRAM) veya CPU üzerinde `yolo11s-pose.pt` temel modeliyle
eğitir ve en iyi ağırlığı `models/football_pitch_pose.pt` konumuna kopyalar.

Güvenlik ve Windows Bellek Kuralları:
    1) API anahtarı YALNIZCA `ROBOFLOW_API_KEY` ortam değişkeninden okunur
       (`--api-key` komutu güvenlik nedeniyle kabul edilmez, komut geçmişine yazılmaz):
           $env:ROBOFLOW_API_KEY = "ANAHTARINIZ"
           $env:YOLO_CONFIG_DIR = "$env:TEMP\\ultralytics_cfg"
           python utils/train_pitch_keypoints.py
       İşlem bitince ortam değişkenini temizleyin:
           Remove-Item Env:ROBOFLOW_API_KEY -ErrorAction SilentlyContinue

    2) Veri seti ve Ultralytics eğitim çıktıları (`runs/`) depo klasörünü
       kirletmemek için varsayılan olarak `%TEMP%\\football_pitch_pose_training`
       altında tutulur. Eğitim sonrası `--cleanup` ile silebilirsiniz.

    3) Varsayılan eğitim ayarları (`yolo11s-pose.pt`, `imgsz=640`, `batch=4`,
       `epochs=60`, `patience=15`, `cache=False`, `amp=True`, `workers<=2`)
       6 GB VRAM ve 16 GB Windows RAM ortamında OOM yaşanmaması için seçilmiştir.
"""

import argparse
import ctypes
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import config  # noqa: E402

# NumPy 2.0+ sürümlerinde kaldırılan np.trapz için Ultralytics 8.3.x uyumluluk yaması
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

DEFAULT_WORK_DIR: str = os.path.join(tempfile.gettempdir(), "football_pitch_pose_training")
DEFAULT_NUM_KEYPOINTS: int = len(getattr(config, "PITCH_KEYPOINTS_32_METERS", ())) or 32
DEFAULT_BASE_MODEL: str = "yolo11s-pose.pt"


def _sanitize_secret(text: str, secret: Optional[str]) -> str:
    """Hata veya log metinlerinden gizli API anahtarını maskeler."""
    if not text or not secret:
        return text
    return text.replace(secret, "***")


def _get_available_ram_gb() -> Optional[float]:
    """Windows üzerinde kullanılabilir fiziksel RAM miktarını GB cinsinden döndürür."""
    if os.name != "nt":
        return None

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None
    return None


def resolve_api_key() -> str:
    """
    Roboflow API anahtarını YALNIZCA ROBOFLOW_API_KEY ortam değişkeninden okur.
    Boş veya yalnızca boşluk içeren anahtarları reddeder.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY ortam değişkeni tanımlı değil. Güvenlik için API anahtarını "
            "yalnızca $env:ROBOFLOW_API_KEY ortam değişkeni üzerinden sağlayın."
        )
    return api_key


def select_dataset_version(project, requested_version: Optional[int] = None):
    """
    İstenen Roboflow veri seti versiyonunu veya belirtilmemişse sayısal olarak
    en yüksek (son) versiyonu deterministik şekilde seçer.
    """
    if requested_version is not None:
        return project.version(int(requested_version))

    versions = project.versions()
    if not versions:
        raise RuntimeError("Roboflow projesinde indirilebilir veri seti versiyonu bulunamadı.")

    def _version_num(ver_obj) -> int:
        try:
            return int(str(ver_obj.version).strip())
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"Geçersiz Roboflow versiyon numarası: {getattr(ver_obj, 'version', ver_obj)}"
            ) from exc

    latest = max(versions, key=_version_num)
    return project.version(_version_num(latest))


def validate_pose_data_yaml(
    data_yaml_path: Union[str, Path],
    expected_num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> Dict[str, Any]:
    """
    YOLO-Pose eğitim veri setinin `data.yaml` dosyasını doğrular:
      - `names` alanı boş olmamalı (`list`/`tuple` veya `dict`)
      - `kpt_shape` `[num_kpts, dims]` biçiminde olmalı (`num_kpts == expected_num_keypoints >= 4`, `dims in (2, 3)`)
      - Varsa `flip_idx`, `0..num_kpts-1` aralığının geçerli bir permütasyonu olmalı
    """
    if (
        not isinstance(expected_num_keypoints, int)
        or isinstance(expected_num_keypoints, bool)
        or expected_num_keypoints < 4
    ):
        raise ValueError(
            f"Beklenen anahtar nokta sayısı en az 4 olan bir tam sayı olmalıdır: {expected_num_keypoints!r}"
        )

    yaml_path = Path(data_yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    data_yaml = {} if loaded is None else loaded
    if not isinstance(data_yaml, dict):
        raise ValueError(f"data.yaml içeriği geçerli bir YAML sözlüğü değil: {yaml_path}")

    raw_names = data_yaml.get("names")
    if isinstance(raw_names, dict):
        if not raw_names:
            raise ValueError(f"data.yaml 'names' sözlüğü boş: {yaml_path}")
        parsed_items = []
        for raw_k, raw_v in raw_names.items():
            if isinstance(raw_k, bool):
                raise ValueError(f"data.yaml 'names' anahtarı boolean olamaz: {raw_k!r}")
            try:
                cid = int(raw_k)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"data.yaml 'names' anahtarı tam sayı olmalıdır: {raw_k!r}"
                ) from exc
            cname = "" if raw_v is None else str(raw_v).strip()
            if not cname:
                raise ValueError(f"data.yaml içinde boş sınıf adı bulundu (id={cid}): {yaml_path}")
            parsed_items.append((cid, cname))
        id_to_name = dict(sorted(parsed_items, key=lambda kv: kv[0]))
    elif isinstance(raw_names, (list, tuple)):
        if not raw_names:
            raise ValueError(f"data.yaml 'names' listesi boş: {yaml_path}")
        id_to_name = {}
        for idx, raw_v in enumerate(raw_names):
            cname = "" if raw_v is None else str(raw_v).strip()
            if not cname:
                raise ValueError(f"data.yaml içinde boş sınıf adı bulundu (index={idx}): {yaml_path}")
            id_to_name[idx] = cname
    else:
        raise ValueError(f"data.yaml içinde geçerli 'names' alanı bulunamadı: {yaml_path}")

    raw_kpt_shape = data_yaml.get("kpt_shape")
    if not isinstance(raw_kpt_shape, (list, tuple)) or len(raw_kpt_shape) != 2:
        raise ValueError(
            f"data.yaml içinde geçerli 'kpt_shape: [N, D]' alanı bulunamadı: {raw_kpt_shape!r}"
        )

    num_kpts, kpt_dims = raw_kpt_shape[0], raw_kpt_shape[1]
    if (
        not isinstance(num_kpts, int)
        or isinstance(num_kpts, bool)
        or not isinstance(kpt_dims, int)
        or isinstance(kpt_dims, bool)
    ):
        raise ValueError(
            f"data.yaml 'kpt_shape' değerleri boolean olmayan tam sayılar olmalıdır: {raw_kpt_shape!r}"
        )

    if num_kpts < 4 or num_kpts != expected_num_keypoints:
        raise ValueError(
            f"Beklenen anahtar nokta sayısı {expected_num_keypoints} iken "
            f"data.yaml kpt_shape={raw_kpt_shape!r} bulundu."
        )

    if kpt_dims not in (2, 3):
        raise ValueError(
            f"data.yaml 'kpt_shape' boyut sayısı 2 veya 3 olmalıdır, bulunan: {kpt_dims!r}"
        )

    flip_idx_out: Optional[list[int]] = None
    if "flip_idx" in data_yaml and data_yaml["flip_idx"] is not None:
        raw_flip = data_yaml["flip_idx"]
        if not isinstance(raw_flip, (list, tuple)) or len(raw_flip) != num_kpts:
            raise ValueError(
                f"data.yaml 'flip_idx' uzunluğu {num_kpts} olmalıdır, bulunan: {raw_flip!r}"
            )
        parsed_flip: list[int] = []
        for idx_val in raw_flip:
            if not isinstance(idx_val, int) or isinstance(idx_val, bool):
                raise ValueError(
                    f"data.yaml 'flip_idx' elemanları tam sayı olmalıdır: {idx_val!r}"
                )
            if idx_val < 0 or idx_val >= num_kpts:
                raise ValueError(
                    f"data.yaml 'flip_idx' indeksi [0, {num_kpts - 1}] aralığı dışında: {idx_val}"
                )
            parsed_flip.append(int(idx_val))
        if len(set(parsed_flip)) != num_kpts:
            raise ValueError(
                f"data.yaml 'flip_idx' tekrar eden indeksler içeremez: {parsed_flip}"
            )
        flip_idx_out = parsed_flip

    return {
        "names": id_to_name,
        "kpt_shape": [int(num_kpts), int(kpt_dims)],
        "flip_idx": flip_idx_out,
    }


def resolve_temp_paths(
    work_dir: Optional[Union[str, Path]],
    project_slug: str,
    version_str: Union[str, int],
) -> Tuple[Path, Path, Path]:
    """
    Veri seti ve Ultralytics eğitim koşusu klasörlerini çalışma dizini (repo)
    dışında, geçici kök (`work_dir`) altında çözümler.
    """
    root = Path(work_dir if work_dir else DEFAULT_WORK_DIR).resolve()
    dataset_dir = root / "datasets" / f"{project_slug}-v{version_str}"
    runs_dir = root / "runs"
    return root, dataset_dir, runs_dir


def resolve_best_weights_path(
    model,
    runs_project: Union[str, Path],
    run_name: str,
) -> Path:
    """
    Eğitim sonrasında Ultralytics trainer nesnesinin gerçek `save_dir` yolundan
    (veya `trainer.best` yedeğinden) `best.pt` dosyasını bulur.
    """
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(runs_project) / run_name))
    best_weights_path = save_dir / "weights" / "best.pt"
    if best_weights_path.is_file():
        return best_weights_path

    trainer_best = getattr(trainer, "best", None)
    if trainer_best:
        candidate = Path(trainer_best)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Beklenen en iyi ağırlık dosyası bulunamadı: {best_weights_path}"
    )


def _resolve_workers(requested_workers: Optional[int]) -> int:
    """Windows düşük RAM koruması için DataLoader worker sayısını belirler."""
    if requested_workers is None:
        avail_ram = _get_available_ram_gb()
        return 0 if (avail_ram is not None and avail_ram < 4.0) else 2
    if os.name == "nt":
        return max(0, min(int(requested_workers), 2))
    return max(0, int(requested_workers))


def _compute_file_sha256(file_path: Path) -> str:
    """Dosyanın SHA-256 özetini parçalı okuma ile hesaplar."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# A2-7: Nokta sırası görsel kontrolü ve yatay çevirme (flip) politikası
# ----------------------------------------------------------------------
ORDER_CHECK_DIR: Path = Path(getattr(config, "BASE_DIR", os.getcwd())) / "runs" / "pitch_order_check"


def template_flip_idx(
    template: Optional[Sequence[Tuple[float, float]]] = None,
    pitch_length: Optional[float] = None,
    tol: float = 1e-3,
) -> Optional[list]:
    """
    Bizim 32 nokta şablonumuzdan yatay çevirme eşlemesini üretir: i. noktanın aynası
    (L - x, y) olan j. nokta. Aynası bulunamayan nokta varsa None döner.
    """
    pts = np.asarray(
        template if template is not None else getattr(config, "PITCH_KEYPOINTS_32_METERS", ()),
        dtype=np.float64,
    )
    length = float(pitch_length if pitch_length is not None else getattr(config, "PITCH_LENGTH_M", 105.0))
    out = []
    for x, y in pts:
        d = np.hypot(pts[:, 0] - (length - x), pts[:, 1] - y)
        j = int(np.argmin(d))
        if float(d[j]) > tol:
            return None
        out.append(j)
    return out if sorted(out) == list(range(len(pts))) else None


def resolve_flip_policy(
    pose_meta: Dict[str, Any],
    data_yaml_path: Union[str, Path],
    fliplr: float = 0.5,
) -> Dict[str, Any]:
    """
    YOLO-Pose yatay çevirme artırması nokta sırasını bozmasın diye karar verir:
      - veri setinde flip_idx var ve şablonla aynı     -> "dataset_ok"
      - veri setinde flip_idx var ama şablondan farklı -> "dataset_mismatch" (fliplr=0 + uyarı:
        nokta sırası config'teki adlarla uyuşmuyor olabilir)
      - flip_idx yok, fliplr > 0, şablondan üretilebiliyor -> "generated" (data.yaml'a yazılır)
      - flip_idx yok, üretilemiyor -> "disabled" (fliplr=0)
      - fliplr <= 0 -> "not_needed"
    """
    tmpl = template_flip_idx()
    ds_flip = pose_meta.get("flip_idx")
    if ds_flip is not None:
        if tmpl is not None and list(ds_flip) != list(tmpl):
            return {
                "action": "dataset_mismatch",
                "fliplr": 0.0,
                "flip_idx": list(ds_flip),
                "message": (
                    "UYARI: Veri setindeki flip_idx bizim 32 nokta şablonumuzdan türetilen eşlemeyle "
                    "uyuşmuyor; nokta sırası config.PITCH_KEYPOINT_NAMES_32 ile farklı olabilir. "
                    "Yatay çevirme kapatıldı (fliplr=0). Eğitimden önce pitch_order_check "
                    "görsellerini mutlaka kontrol edin."
                ),
            }
        return {"action": "dataset_ok", "fliplr": float(fliplr), "flip_idx": list(ds_flip),
                "message": "Veri setindeki flip_idx şablonla uyumlu."}
    if fliplr <= 0:
        return {"action": "not_needed", "fliplr": 0.0, "flip_idx": None,
                "message": "Yatay çevirme kapalı; flip_idx gerekmiyor."}
    if tmpl is None:
        return {"action": "disabled", "fliplr": 0.0, "flip_idx": None,
                "message": "UYARI: flip_idx yok ve şablondan üretilemedi; yatay çevirme kapatıldı (fliplr=0)."}
    yaml_path = Path(data_yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["flip_idx"] = list(tmpl)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return {"action": "generated", "fliplr": float(fliplr), "flip_idx": list(tmpl),
            "message": "flip_idx veri setinde yoktu; bizim şablondan üretilip data.yaml'a yazıldı."}


def read_pose_label(
    label_path: Union[str, Path], img_w: int, img_h: int, num_kpts: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """YOLO-Pose etiket dosyasındaki ilk örneğin noktalarını piksel olarak okur (xy, görünürlük)."""
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            if len(vals) < 5 + 2 * num_kpts:
                continue
            arr = np.asarray([float(v) for v in vals[5:]], dtype=np.float64)
            dims = 3 if arr.size >= 3 * num_kpts else 2
            arr = arr[: dims * num_kpts].reshape(num_kpts, dims)
            xy = arr[:, :2] * np.array([img_w, img_h], dtype=np.float64)
            vis = arr[:, 2] > 0 if dims == 3 else np.ones(num_kpts, dtype=bool)
            vis &= (arr[:, 0] > 0) | (arr[:, 1] > 0)
            return xy, vis
    return None


def draw_keypoint_order(image: np.ndarray, xy: np.ndarray, vis: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Görünür noktaları indeks ve BİZİM config adımızla çizer."""
    import cv2

    out = image.copy()
    for i, ((x, y), v) in enumerate(zip(xy, vis)):
        if not v:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 6, (0, 0, 255), -1)
        label = f"{i}:{names[i] if i < len(names) else '?'}"
        cv2.putText(out, label, (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, label, (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def write_order_check_images(
    dataset_dir: Union[str, Path],
    out_dir: Union[str, Path] = ORDER_CHECK_DIR,
    names: Optional[Sequence[str]] = None,
    num_kpts: int = DEFAULT_NUM_KEYPOINTS,
    count: int = 3,
) -> list:
    """Veri setinden en çok görünür noktası olan `count` görüntüyü adlarımızla çizip PNG yazar."""
    import cv2

    names = list(names if names is not None else getattr(config, "PITCH_KEYPOINT_NAMES_32", ()))
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    scored = []
    for label_path in sorted(dataset_dir.glob("**/labels/*.txt")):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = label_path.parent.parent / "images" / (label_path.stem + ext)
            if cand.is_file():
                img_path = cand
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        parsed = read_pose_label(label_path, img.shape[1], img.shape[0], num_kpts)
        if parsed is None:
            continue
        scored.append((int(parsed[1].sum()), str(label_path), img_path, parsed))
        if len(scored) >= 50:
            break
    scored.sort(key=lambda t: (-t[0], t[1]))
    if scored:
        out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, (_n, _lp, img_path, (xy, vis)) in enumerate(scored[:count], start=1):
        img = cv2.imread(str(img_path))
        out_path = out_dir / f"order_check_{k}_{img_path.stem[:40]}.png"
        cv2.imwrite(str(out_path), draw_keypoint_order(img, xy, vis, names))
        written.append(out_path)
    return written


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "32 noktalı saha anahtar nokta (YOLO-Pose) kalibrasyon modelini eğitir. "
            "Roboflow API anahtarı yalnızca ROBOFLOW_API_KEY ortam değişkeninden okunur."
        )
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="roboflow-jvuqo",
        help="Roboflow workspace slug'ı.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="football-field-detection-f07vi",
        help="Roboflow saha anahtar nokta (keypoint) proje slug'ı.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow veri seti versiyon numarası. Boş bırakılırsa en son versiyon seçilir.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=(
            "Başlangıç YOLO-Pose ağırlığı (varsayılan: yolo11s-pose.pt; "
            "6 GB RTX 4050 VRAM için güvenli seçim)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60, help="Eğitim epoch sayısı.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(getattr(config, "PITCH_KEYPOINT_IMGSZ", 640)),
        help="Eğitim görüntü boyutu (piksel, varsayılan: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch boyutu (6 GB VRAM için güvenli varsayılan: 4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker sayısı (otomatik: düşük RAM <4GB ise 0, aksi halde <=2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Eğitim cihazı ('0', 'cuda' veya 'cpu'; boşsa otomatik CUDA/CPU seçilir).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Erken durdurma (early stopping) sabır epoch sayısı.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="football_pitch_pose",
        help="Eğitim koşusu alt klasör adı.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Eğitim sonrası en iyi ağırlığın kopyalanacağı hedef .pt yolu "
            "(varsayılan: config.PITCH_KEYPOINT_MODEL_PATH -> models/football_pitch_pose.pt)."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=DEFAULT_WORK_DIR,
        help="Veri seti ve Ultralytics runs/ için geçici kök dizin (varsayılan: %%TEMP%%).",
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Geçici dizindeki veri setini yeniden indirir.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Eğitim bitince geçici veri seti ve run klasörlerini siler.",
    )
    parser.add_argument(
        "--check-order-only",
        action="store_true",
        help=(
            "Veri setini indirip doğrular, nokta sırası kontrol görsellerini "
            "runs/pitch_order_check/ altına yazar ve eğitime BAŞLAMADAN çıkar."
        ),
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Yatay çevirme artırma olasılığı (flip_idx yoksa şablondan üretilir).",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help=(
            "Mosaic artırma olasılığı. Sahanın kısmi görüldüğü (TV yakın çekim) karelere "
            "genelleme sağlar; tamamen kapatmak (0) TV görüntülerinde tespiti çökertir."
        ),
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=40,
        help=(
            "Son N epoch'ta mosaic kapatılır. Nokta hassasiyeti bu evrede oturur; "
            "Ultralytics varsayılanı (10) saha noktaları için yetersiz kaldı."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.7,
        help="Ölçek artırma aralığı (±). Yakınlaştırılmış yayın karelerine genelleme için.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(tempfile.gettempdir(), "ultralytics_cfg"),
    )

    try:
        api_key = resolve_api_key()
    except ValueError as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)

    try:
        import torch
        from roboflow import Roboflow
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "\n[HATA] Gerekli paketler yüklü değil (ultralytics, roboflow, torch).\n"
            f"Orijinal hata: {exc}\n"
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # 1) VERİ SETİNİ GEÇİCİ DİZİNE İNDİR (VEYA YEREL ÖNBELLEĞİ KULLAN)
    # --------------------------------------------------------------
    work_root = Path(args.work_dir if args.work_dir else DEFAULT_WORK_DIR).resolve()
    cached_yaml_candidates = sorted(
        work_root.glob(f"datasets/{args.project}-v*/data.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    use_local_cache = bool(
        cached_yaml_candidates
        and not args.overwrite_dataset
        and args.version is None
        and getattr(Roboflow, "__module__", "").startswith("roboflow")
    )
    if use_local_cache:
        cached_dir = cached_yaml_candidates[0].parent
        work_dir = work_root
        runs_dir = work_root / "runs"
        data_yaml_path = str(cached_yaml_candidates[0])
        dataset_location = str(cached_dir)
        print(f"[1/4] Önbellekteki hazır saha veri seti kullanılıyor: {dataset_location}")
    else:
        print(
            f"[1/4] Roboflow'a bağlanılıyor (workspace={args.workspace}, "
            f"project={args.project})..."
        )
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(args.workspace).project(args.project)
            version = select_dataset_version(project, args.version)
            print(f"      Seçilen veri seti versiyonu: {version.version}")

            work_dir, dataset_dir, runs_dir = resolve_temp_paths(
                args.work_dir, args.project, version.version
            )
            dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            overwrite_dl = bool(
                args.overwrite_dataset
                or (dataset_dir.exists() and not (dataset_dir / "data.yaml").is_file())
            )
            dataset = version.download("yolov8", location=str(dataset_dir), overwrite=overwrite_dl)
            dataset_location = str(dataset.location)
        except Exception as exc:
            safe_msg = _sanitize_secret(str(exc), api_key)
            print(f"[HATA] Roboflow saha veri seti indirilemedi: {safe_msg}")
            sys.exit(1)

        data_yaml_path = os.path.join(dataset_location, "data.yaml")
        print(f"      Saha veri seti hazır: {dataset_location}")

    # --------------------------------------------------------------
    # 2) data.yaml ANAHTAR NOKTA (POSE) YAPISINI DOĞRULA
    # --------------------------------------------------------------
    try:
        pose_meta = validate_pose_data_yaml(
            data_yaml_path, expected_num_keypoints=DEFAULT_NUM_KEYPOINTS
        )
    except Exception as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)

    print("\n[2/4] Doğrulanan YOLO-Pose veri seti şeması:")
    print(f"      Sınıflar : {pose_meta['names']}")
    print(f"      kpt_shape: {pose_meta['kpt_shape']}")
    if pose_meta["flip_idx"] is not None:
        print(f"      flip_idx : {len(pose_meta['flip_idx'])} elemanlı geçerli permütasyon")
    if pose_meta["kpt_shape"] != [DEFAULT_NUM_KEYPOINTS, 3]:
        print(
            f"[HATA] Beklenen kpt_shape [{DEFAULT_NUM_KEYPOINTS}, 3], bulunan {pose_meta['kpt_shape']}."
        )
        sys.exit(1)

    flip_policy = resolve_flip_policy(pose_meta, data_yaml_path, fliplr=args.fliplr)
    print(
        f"      flip     : {flip_policy['action']} (fliplr={flip_policy['fliplr']}) - "
        f"{flip_policy['message']}"
    )

    order_pngs = write_order_check_images(dataset_location, ORDER_CHECK_DIR)
    if order_pngs:
        print("      Nokta sırası kontrol görselleri (adlar config.PITCH_KEYPOINT_NAMES_32'den):")
        for pth in order_pngs:
            print(f"        {pth}")
    else:
        print("[UYARI] Nokta sırası kontrol görseli üretilemedi (etiketli görüntü bulunamadı).")
    if args.check_order_only:
        print(
            "\n[DURDURULDU] --check-order-only: görsellerdeki adların noktaların gerçek yerleriyle "
            "eşleştiğini kontrol edin; doğruysa aynı komutu bu bayrak olmadan çalıştırıp eğitimi başlatın."
        )
        return

    # --------------------------------------------------------------
    # 3) RTX 4050 GÜVENLİ YOLO-POSE EĞİTİMİNİ BAŞLAT
    # --------------------------------------------------------------
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    workers = _resolve_workers(args.workers)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_project = str(runs_dir)

    print(
        f"\n[3/4] Saha kalibrasyon modeli eğitimi başlıyor: base_model={args.base_model}, "
        f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={device}, workers={workers}, cache=False, amp=True"
    )
    print(f"      Geçici run dizini: {runs_project}")

    model = YOLO(args.base_model)
    model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=workers,
        cache=False,
        amp=True,
        project=runs_project,
        name=args.run_name,
        patience=args.patience,
        save=True,
        plots=True,
        fliplr=flip_policy["fliplr"],
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
    )

    # --------------------------------------------------------------
    # 4) EN İYİ AĞIRLIKLARI HEDEF KONUMA KOPYALA
    # --------------------------------------------------------------
    try:
        best_weights_path = resolve_best_weights_path(model, runs_project, args.run_name)
    except FileNotFoundError as exc:
        print(f"\n[HATA] {exc}")
        sys.exit(1)

    destination_path = (
        Path(args.output).resolve()
        if args.output
        else Path(config.PITCH_KEYPOINT_MODEL_PATH).resolve()
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_path, destination_path)
    weights_sha256 = _compute_file_sha256(destination_path)

    print(f"\n[4/4] Eğitim tamamlandı! En iyi saha modeli kaydedildi: {destination_path}")
    print(f"      SHA-256 özeti: {weights_sha256}")

    print("\n" + "=" * 70)
    print("SONRAKİ ADIM: Otomatik saha kalibrasyonu için doğrulama bilgileri:")
    print("=" * 70)
    print(f"PITCH_KEYPOINT_MODEL_PATH -> {destination_path}")
    print(f'$env:FOOTBALL_PITCH_MODEL_SHA256 = "{weights_sha256}"')
    print("Ortam değişkeni yerine config/config.py'de sabitlemek için:")
    print(f'SECURITY_PITCH_MODEL_SHA256 = "{weights_sha256}"')
    print("=" * 70)

    save_dir = best_weights_path.parent.parent
    if args.cleanup:
        for target in (Path(dataset_location).resolve(), save_dir.resolve()):
            try:
                target.relative_to(work_dir)
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
            except ValueError:
                pass
        print(f"[TEMİZLİK] Geçici dizinler temizlendi ({work_dir}).")
    else:
        print(
            "\n[BİLGİ] Geçici veri seti ve eğitim çıktıları doğrulama için korundu:\n"
            f"        Veri seti : {dataset_location}\n"
            f"        Run dizini: {save_dir}\n"
            "        Doğrulama sonrası --cleanup veya manuel silme ile kaldırabilirsiniz."
        )


if __name__ == "__main__":
    main()
