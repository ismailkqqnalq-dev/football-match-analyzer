"""
train_custom_yolo.py
----------------------
FUTBOLA ÖZEL (4 SINIFLI) YOLO MODELİ EĞİTİM BETİĞİ

Bu betik, Roboflow üzerindeki `football-players-detection` veri setini
(`ball`, `goalkeeper`, `player`, `referee`) geçici dizin (`%TEMP%`) altına
indirip yerel NVIDIA GPU (ör. RTX 4050 6 GB VRAM) veya CPU üzerinde
fine-tune eder ve en iyi ağırlığı `models/` dizinine kopyalar.

Güvenlik ve Windows Bellek Kuralları:
    1) API anahtarını komut satırında `--api-key` ile vermek yerine geçici
       ortam değişkeni olarak tanımlayın (komut geçmişine yazılmaz):
           $env:ROBOFLOW_API_KEY = "ANAHTARINIZ"
           $env:YOLO_CONFIG_DIR = "$env:TEMP\\ultralytics_cfg"
           python utils/train_custom_yolo.py
       İşlem bitince ortam değişkenini temizleyin:
           Remove-Item Env:ROBOFLOW_API_KEY -ErrorAction SilentlyContinue

    2) Veri seti ve Ultralytics eğitim çıktıları (`runs/`) depo klasörünü
       kirletmemek için varsayılan olarak `%TEMP%\\football_yolo_training`
       altında tutulur. İsterseniz eğitim sonrası `--cleanup` ile silebilirsiniz.

    3) Varsayılan eğitim ayarları (`yolo11s.pt`, `imgsz=960`, `batch=4`,
       `epochs=60`, `patience=15`, `cache=False`, `amp=True`, `workers<=2`)
       6 GB VRAM ve 16 GB Windows RAM ortamında OOM yaşanmaması için seçilmiştir.
"""

import argparse
import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

# NumPy 2.0+ sürümlerinde kaldırılan np.trapz için Ultralytics 8.3.x uyumluluk yaması
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

EXPECTED_CLASSES: Set[str] = {"ball", "goalkeeper", "player", "referee"}
DEFAULT_WORK_DIR: str = os.path.join(tempfile.gettempdir(), "football_yolo_training")


def _sanitize_secret(text: str, secret: Optional[str]) -> str:
    """Hata veya log metinlerinden gizli API anahtarını maskeler."""
    if not text or not secret:
        return text
    sanitized = text.replace(secret, "***")
    return sanitized


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


def resolve_api_key(cli_api_key: Optional[str] = None, warn_fn=print) -> str:
    """
    Roboflow API anahtarını öncelikle ROBOFLOW_API_KEY ortam değişkeninden okur;
    yalnızca ortam değişkeni yoksa `--api-key` argümanına düşer.
    Boş veya yalnızca boşluk içeren anahtarları reddeder.
    """
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    cli_key = (cli_api_key or "").strip()

    if cli_key and not env_key and warn_fn is not None:
        warn_fn(
            "[UYARI] Güvenlik için --api-key yerine ROBOFLOW_API_KEY ortam "
            "değişkenini kullanmanız önerilir."
        )

    api_key = env_key or cli_key
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY ortam değişkeni tanımlı değil ve geçerli bir --api-key verilmedi."
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


def validate_data_yaml(
    data_yaml_path: str | Path,
    expected_classes: Optional[Set[str]] = None,
) -> Dict[int, str]:
    """
    data.yaml içindeki sınıf haritasını (`list` veya `dict`) normalize eder ve
    beklenen 4 futbol sınıfı (`ball`, `goalkeeper`, `player`, `referee`) ile
    birebir eşleştiğini doğrular.
    """
    yaml_path = Path(data_yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f) or {}

    raw_names = data_yaml.get("names")
    if isinstance(raw_names, dict):
        if not raw_names:
            raise ValueError(f"data.yaml 'names' sözlüğü boş: {yaml_path}")
        id_to_name = {
            int(k): str(v).strip()
            for k, v in sorted(raw_names.items(), key=lambda kv: int(kv[0]))
        }
    elif isinstance(raw_names, (list, tuple)):
        if not raw_names:
            raise ValueError(f"data.yaml 'names' listesi boş: {yaml_path}")
        id_to_name = {i: str(v).strip() for i, v in enumerate(raw_names)}
    else:
        raise ValueError(f"data.yaml içinde geçerli 'names' alanı bulunamadı: {yaml_path}")

    expected = expected_classes if expected_classes is not None else EXPECTED_CLASSES
    found = set(id_to_name.values())
    if found != expected:
        raise ValueError(
            f"Beklenen sınıflar {sorted(expected)} iken data.yaml sınıfları {id_to_name} bulundu."
        )

    return id_to_name


def resolve_temp_paths(
    work_dir: Optional[str | Path],
    project_slug: str,
    version_str: str | int,
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
    runs_project: str | Path,
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


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Futbola özel 4 sınıflı YOLO modeli eğitir."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        default=None,
        help=(
            "Roboflow Private API Key (isteğe bağlı yedek). Güvenlik için "
            "ROBOFLOW_API_KEY ortam değişkeni tercih edilmelidir."
        ),
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
        default="football-players-detection-3zvbc",
        help="Roboflow proje slug'ı.",
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
        default="yolo11s.pt",
        help=(
            "Başlangıç COCO ağırlığı. 6 GB RTX 4050 ve 90 dk maç hızı için "
            "'yolo11s.pt' veya 'yolov8s.pt' önerilir; 'yolov8m.pt' 960px'te VRAM riski taşır."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60, help="Eğitim epoch sayısı.")
    parser.add_argument("--imgsz", type=int, default=960, help="Eğitim görüntü boyutu (piksel).")
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
        default="football_finetune",
        help="Eğitim koşusu alt klasör adı.",
    )
    parser.add_argument(
        "--output-model-name",
        type=str,
        default="football_yolo11s_best.pt",
        help="models/ klasörüne kopyalanacak çıktı ağırlık dosyası adı.",
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(tempfile.gettempdir(), "ultralytics_cfg"),
    )

    try:
        api_key = resolve_api_key(args.api_key)
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
        print(f"[1/4] Önbellekteki hazır Roboflow veri seti kullanılıyor: {dataset_location}")
    else:
        print(f"[1/4] Roboflow'a bağlanılıyor (workspace={args.workspace}, project={args.project})...")
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
            print(f"[HATA] Roboflow veri seti indirilemedi: {safe_msg}")
            sys.exit(1)

        data_yaml_path = os.path.join(dataset_location, "data.yaml")
        print(f"      Veri seti hazır: {dataset_location}")

    # --------------------------------------------------------------
    # 2) data.yaml SINIFLARINI DOĞRULA
    # --------------------------------------------------------------
    try:
        id_to_name = validate_data_yaml(data_yaml_path)
    except Exception as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)

    print("\n[2/4] Doğrulanan sınıf eşlemesi (ID -> name):")
    for cid, cname in id_to_name.items():
        print(f"      {cid}: {cname}")

    # --------------------------------------------------------------
    # 3) RTX 4050 GÜVENLİ EĞİTİMİ BAŞLAT
    # --------------------------------------------------------------
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    workers = _resolve_workers(args.workers)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_project = str(runs_dir)

    print(
        f"\n[3/4] Eğitim başlıyor: base_model={args.base_model}, epochs={args.epochs}, "
        f"imgsz={args.imgsz}, batch={args.batch}, device={device}, workers={workers}, "
        f"cache=False, amp=True"
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
    )

    # --------------------------------------------------------------
    # 4) EN İYİ AĞIRLIKLARI models/ DİZİNİNE KOPYALA
    # --------------------------------------------------------------
    try:
        best_weights_path = resolve_best_weights_path(model, runs_project, args.run_name)
    except FileNotFoundError as exc:
        print(f"\n[HATA] {exc}")
        sys.exit(1)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    destination_path = models_dir / args.output_model_name
    shutil.copy2(best_weights_path, destination_path)

    print(f"\n[4/4] Eğitim tamamlandı! En iyi ağırlık kaydedildi: {destination_path}")

    target_class_ids_str = "{" + ", ".join(f'{cid}: "{cname}"' for cid, cname in id_to_name.items()) + "}"
    print("\n" + "=" * 70)
    print("SONRAKİ ADIM: config/config.py içinde şu iki ayarı doğrulayın:")
    print("=" * 70)
    print(f'YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "{args.output_model_name}")')
    print(f"TARGET_CLASS_IDS = {target_class_ids_str}")
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
