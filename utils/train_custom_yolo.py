"""
train_custom_yolo.py
----------------------
FİNE-TUNE EDİLMİŞ (FUTBOLA ÖZEL) YOLO MODELİ EĞİTİM BETİĞİ

⚠️  ÖNEMLİ -- BU BETİĞİ NEREDE ÇALIŞTIRMALISIN:
    Bu betik BU CHAT ORTAMINDA (Claude'un sanal alanında) ÇALIŞTIRILAMAZ --
    burada internet erişimi ve GPU yok. Bu dosyayı KENDİ bilgisayarına ya da
    Google Colab / Kaggle Notebooks gibi ÜCRETSİZ GPU sağlayan bir ortama
    kopyalayıp orada çalıştırman gerekiyor. Aşağıdaki adımlar bunun için.

NEDEN K-MEANS/DEĞİL DE FINE-TUNE?
    Şu ana kadarki pipeline, standart (COCO ile eğitilmiş) YOLOv8'in sadece
    "person" ve "sports ball" sınıflarını tanıdığını, hakem/oyuncu ayrımını
    ise K-Means forma renk kümelemesiyle SONRADAN tahmin ettiğimizi
    varsayıyordu. Bu çalışır ama iki sınırlaması var:
      1) Top küçük/hızlı olduğu için COCO modeli onu sık sık kaçırır
         (COCO'da "sports ball" sınıfı basketbol/tenis topu gibi çok
         farklı nesneleri de kapsadığından, futbol topuna özel değildir).
      2) Hakem/oyuncu ayrımı K-Means'in "en az üyeli küme" sezgisine bağlıdır
         -- bu sezgi çoğu zaman doğru çalışır ama garantili değildir.
    Fine-tune edilmiş bir model, bu ayrımları YOLO SEVİYESİNDE, doğrudan
    öğrenerek yapar -- hem daha isabetli top tespiti hem de %100 güvenilir
    hakem/oyuncu/kaleci ayrımı sağlar.

ADIM ADIM SÜREÇ:
    1) KENDİ GPU'LU ORTAMINI HAZIRLA
       Google Colab (ücretsiz, önerilen): https://colab.research.google.com
       Colab'da Çalışma Zamanı -> Çalışma zamanı türünü değiştir -> GPU (T4) seç.

    2) GEREKLİ PAKETLERİ KUR (Colab hücresinde ya da terminalde)
           pip install ultralytics roboflow

    3) ROBOFLOW API KEY AL (ücretsiz hesap yeterli)
       https://app.roboflow.com adresine kayıt ol -> sağ üstten hesap
       ayarlarına git -> "Roboflow API" sekmesinden Private API Key'ini kopyala.

    4) VERİ SETİNİ İNDİR
       Bu betik, Roboflow Universe'teki halka açık "football-players-detection"
       projesini kullanacak şekilde ayarlı:
           https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
       Sınıfları: ball(0), goalkeeper(1), player(2), referee(3).
       Farklı bir Roboflow projesi/veri seti kullanmak istersen, aşağıdaki
       ROBOFLOW_WORKSPACE / ROBOFLOW_PROJECT / ROBOFLOW_VERSION değerlerini
       kendi projenin sayfasındaki "Download this Dataset" bölümünden kopyala.

    5) BU BETİĞİ ÇALIŞTIR
           python train_custom_yolo.py --api-key SENIN_API_KEYIN --epochs 100

       Eğitim süresi veri seti büyüklüğüne, epoch sayısına ve GPU'ya göre
       değişir -- KABA bir fikir vermesi için: birkaç bin görüntülük bir veri
       setinde, Colab'ın ücretsiz T4 GPU'sunda, 100 epoch tipik olarak
       1-3 saat sürebilir. Bu KESİN bir rakam değil, sadece kabaca planlama
       için bir referans.

    6) SONUCU PROJEYE ENTEGRE ET
       Eğitim bitince en iyi ağırlıklar şurada olacak:
           runs/detect/football_finetune/weights/best.pt
       Bu dosyayı indirip football_analysis_system/models/ klasörüne kopyala,
       sonra config/config.py içinde SADECE şu iki satırı değiştir:

           YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")  # veya verdiğin isim
           TARGET_CLASS_IDS = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

       Bu betiğin sonunda tam olarak hangi TARGET_CLASS_IDS eşlemesini
       kullanman gerektiğini (veri setinin gerçek data.yaml sırasına göre)
       otomatik olarak ekrana yazdırıyoruz -- kopyala-yapıştır yapman yeterli.
       BAŞKA HİÇBİR DOSYAYA (main.py, tracker.py, team_classifier.py, vb.)
       DOKUNMANA GEREK YOK -- pipeline'ın geri kalanı bu iki değeri otomatik
       okuyup kendini ayarlıyor (bkz. config.py içindeki HUMAN_CLASS_NAMES,
       USES_FINE_TUNED_FOOTBALL_MODEL türetmeleri).
"""

import argparse
import os
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Futbola özel fine-tune edilmiş bir YOLOv8 modeli eğitir."
    )
    parser.add_argument(
        "--api-key", type=str, required=True,
        help="Roboflow hesabından aldığın Private API Key.",
    )
    parser.add_argument(
        "--workspace", type=str, default="roboflow-jvuqo",
        help="Roboflow workspace slug'ı (varsayılan: halka açık football-players-detection projesi).",
    )
    parser.add_argument(
        "--project", type=str, default="football-players-detection-3zvbc",
        help="Roboflow proje slug'ı.",
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help=(
            "Roboflow veri seti versiyon numarası. Belirtilmezse en son versiyon "
            "kullanılır -- projenin Roboflow sayfasındaki 'Versions' sekmesinden "
            "teyit edebilirsin (versiyon numaraları zamanla değişebilir)."
        ),
    )
    parser.add_argument(
        "--base-model", type=str, default="yolov8m.pt",
        help=(
            "Fine-tune için başlangıç noktası olacak, COCO ile önceden eğitilmiş "
            "model. 'yolov8n.pt' en hızlı/en küçük, 'yolov8x.pt' en yavaş/en doğru. "
            "'yolov8m.pt' hız/doğruluk dengesi için makul bir varsayılan."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100, help="Eğitim epoch sayısı.")
    parser.add_argument("--imgsz", type=int, default=960, help="Eğitim görüntü boyutu (piksel).")
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch boyutu. GPU belleği yetmezse (CUDA out of memory) küçült (ör. 8 ya da 4).",
    )
    parser.add_argument(
        "--run-name", type=str, default="football_finetune",
        help="Bu eğitim koşusunun ismi (runs/detect/<run-name>/ altına kaydedilir).",
    )
    parser.add_argument(
        "--output-model-name", type=str, default="football_yolov8_best.pt",
        help="Eğitim bitince, en iyi ağırlıkların kopyalanacağı dosya ismi.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Bu importlar fonksiyon içinde yapılıyor ki, betik yanlışlıkla GPU/internet
    # olmayan bir ortamda çalıştırılırsa hata mesajı en baştaki docstring'i
    # okumadan önce kafa karıştırıcı bir import hatasına dönüşmesin -- yine de
    # anlamlı bir mesajla karşılaşman için try/except ile sarıyoruz.
    try:
        from roboflow import Roboflow
        from ultralytics import YOLO
    except ImportError as e:
        print(
            "\n[HATA] Gerekli paketler yüklü değil.\n"
            "Bu betiği çalıştırmadan önce şunu çalıştır:\n"
            "    pip install ultralytics roboflow\n"
            f"\nOrijinal hata: {e}\n"
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # 1) VERİ SETİNİ ROBOFLOW'DAN İNDİR (YOLOv8 formatında)
    # --------------------------------------------------------------
    print(f"[1/4] Roboflow'a bağlanılıyor (workspace={args.workspace}, project={args.project})...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)

    version = project.version(args.version) if args.version else project.version(project.versions()[-1].version)
    print(f"      Veri seti versiyonu: {version.version}")

    dataset = version.download("yolov8")
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    print(f"      Veri seti indirildi: {dataset.location}")

    # --------------------------------------------------------------
    # 2) data.yaml'DAKİ SINIF SIRASINI OKU (config.py'ye kopyalanacak eşleme için)
    # --------------------------------------------------------------
    import yaml  # PyYAML, ultralytics'in bağımlılığı olarak zaten kurulu olur

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get("names", [])

    print(f"\n[2/4] Veri setindeki sınıf sırası: {class_names}")

    # --------------------------------------------------------------
    # 3) EĞİTİMİ BAŞLAT (transfer learning: COCO ağırlıklarından devam)
    # --------------------------------------------------------------
    print(f"\n[3/4] Eğitim başlıyor: base_model={args.base_model}, epochs={args.epochs}, imgsz={args.imgsz}")
    print("      Bu adım GPU'na ve veri seti boyutuna göre uzun sürebilir (dakikalar-saatler).")

    model = YOLO(args.base_model)
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.run_name,
        patience=20,        # 20 epoch boyunca iyileşme olmazsa erken durdur (overfitting'i önler)
        save=True,
        plots=True,          # eğitim/doğrulama grafiklerini (loss, mAP eğrileri) kaydet
    )

    # --------------------------------------------------------------
    # 4) EN İYİ AĞIRLIKLARI PROJEYE KOPYALA
    # --------------------------------------------------------------
    best_weights_path = os.path.join("runs", "detect", args.run_name, "weights", "best.pt")

    if not os.path.exists(best_weights_path):
        print(f"\n[UYARI] Beklenen ağırlık dosyası bulunamadı: {best_weights_path}")
        print("Eğitim loglarını kontrol et -- 'runs/detect/' altında farklı bir isimle kaydedilmiş olabilir.")
        sys.exit(1)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    destination_path = os.path.join(models_dir, args.output_model_name)
    shutil.copy(best_weights_path, destination_path)

    print(f"\n[4/4] Eğitim tamamlandı! En iyi ağırlıklar şuraya kopyalandı: {destination_path}")

    # --------------------------------------------------------------
    # config.py'YE KOPYALANACAK TAM AYARI YAZDIR
    # --------------------------------------------------------------
    target_class_ids_str = "{" + ", ".join(f'{i}: "{name}"' for i, name in enumerate(class_names)) + "}"

    print("\n" + "=" * 70)
    print("SONRAKİ ADIM: config/config.py içinde SADECE şu iki satırı değiştir:")
    print("=" * 70)
    print(f'YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "{args.output_model_name}")')
    print(f"TARGET_CLASS_IDS = {target_class_ids_str}")
    print("=" * 70)
    print(
        "Başka hiçbir dosyaya dokunmana gerek yok -- HUMAN_CLASS_NAMES, "
        "BALL_CLASS_NAME, USES_FINE_TUNED_FOOTBALL_MODEL ve "
        "TEAM_CLASSIFIER_N_CLUSTERS bu iki değerden otomatik türetilir."
    )


if __name__ == "__main__":
    main()
