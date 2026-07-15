"""
config.py
---------
Projedeki TÜM sabitleri (dosya yolları, model parametreleri, eşik değerleri vb.)
tek bir merkezden yönetmek için oluşturulmuş konfigürasyon dosyası.

Neden ayrı bir config dosyası?
- Modüller içine "sihirli sayılar" (magic numbers) gömmek yerine, her ayarı
  buradan değiştirebilmek proje bakımını kolaylaştırır.
- Örneğin YOLO güven eşiğini (confidence threshold) değiştirmek istediğinde
  tüm kod tabanını taramana gerek kalmaz, sadece burayı güncellersin.
"""

import os

# ------------------------------------------------------------------
# TEMEL DİZİN YOLLARI
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "input_videos")
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "output_videos")
EXPORT_DIR = os.path.join(BASE_DIR, "data", "exports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Örnek varsayılan giriş/çıkış video yolları (main.py içinde override edilebilir)
DEFAULT_INPUT_VIDEO = os.path.join(INPUT_VIDEO_DIR, "match_sample.mp4")
DEFAULT_OUTPUT_VIDEO = os.path.join(OUTPUT_VIDEO_DIR, "match_sample_annotated.mp4")

# ------------------------------------------------------------------
# YOLO NESNE TESPİTİ AYARLARI
# ------------------------------------------------------------------
# Not: Standart "yolov8x.pt" COCO veri seti ile eğitilmiştir ve sadece
# "person" (0) ve "sports ball" (32) sınıflarını tanır. Hakem ile oyuncuyu
# YOLO seviyesinde ayırt edemez (ikisi de "person"). Hakem/oyuncu/takım
# ayrımını biz 4. adımda (K-Means forma renk kümeleme) yapacağız.
#
# Daha iyi sonuç için Roboflow üzerinde bulunan hazır "football-players-detection"
# gibi futbola özel fine-tune edilmiş bir YOLO modeli de kullanılabilir
# (bu modeller genelde player/goalkeeper/referee/ball sınıflarını ayrı verir).
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8x.pt")  # veya kendi fine-tune modelin

# Tespit güven eşiği: Bu değerin altındaki tespitler göz ardı edilir.
DETECTION_CONFIDENCE_THRESHOLD = 0.3

# IOU eşiği (Non-Max Suppression için): Örtüşen kutuların birleştirilme oranı.
DETECTION_IOU_THRESHOLD = 0.45

# Bizim ilgilendiğimiz COCO sınıf ID'leri:
#   0  -> person (oyuncu + hakem, henüz ayrılmamış)
#   32 -> sports ball (top)
TARGET_CLASS_IDS = {
    0: "person",
    32: "ball",
}

# Cihaz seçimi: GPU varsa "cuda", yoksa "cpu" (Apple Silicon için "mps")
DEVICE = "cuda"  # gerekirse "cpu" olarak değiştir

# ------------------------------------------------------------------
# TAKİP (TRACKING) AYARLARI - DeepSORT
# ------------------------------------------------------------------
# max_age: Bir nesne tespit edilemeden (örn. kamera dışına çıkınca) kaç FRAME
# boyunca track'in "hayatta" tutulacağını belirler. Bu değer ne kadar büyükse,
# oyuncu ekrandan uzun süre kaybolsa bile geri geldiğinde aynı ID'yi alma
# ihtimali o kadar artar. FAKAT çok büyük tutmak, biri sahadan tamamen çıkıp
# başka biri girdiğinde de eski ID'nin yanlışlıkla yeni oyuncuya atanma riskini
# artırır. Yayın maçlarda tipik değer: 30-90 frame arası (fps=25 varsayımıyla
# ~1.2 - 3.6 saniyeye denk gelir).
TRACKER_MAX_AGE = 70

# n_init: Bir tespitin "gerçek" bir track olarak onaylanması (confirmed) için
# üst üste kaç karede eşleşmesi gerektiği. Küçük tutulursa gürültülü/yanlış
# tespitler bile hemen ID alır; büyük tutulursa gerçek oyuncular bile ID almadan
# önce birkaç kare bekler.
TRACKER_N_INIT = 3

# max_cosine_distance: Görünüm (appearance) embedding'leri arasındaki kosinüs
# mesafesi eşiği. Bu değerin altındaki mesafeler "aynı kişi" olarak kabul edilir.
# Küçültürsen eşleştirme daha katı (daha az yanlış eşleşme, ama ID kaybı artabilir),
# büyütürsen daha esnek (ID sürekliliği artar, ama forma rengi/postür benzer
# oyuncular karışabilir).
TRACKER_MAX_COSINE_DISTANCE = 0.3

# nn_budget: Her track için bellekte tutulacak maksimum geçmiş görünüm
# embedding sayısı. None -> sınırsız (bellek maliyeti artar).
TRACKER_NN_BUDGET = 100

# embedder: Görünüm özelliği (appearance feature) çıkarmak için kullanılan
# hafif CNN modeli. "mobilenet" hızlı ve CPU'da bile makul çalışır.
# (Bu embedder torch/torchvision gerektirir.)
TRACKER_EMBEDDER = "mobilenet"

# GPU varsa embedder'ı hızlandırmak için True yap (torch CUDA gerektirir).
TRACKER_EMBEDDER_GPU = False

# ------------------------------------------------------------------
# TOP TAKİBİ (BALL TRACKING) AYARLARI
# ------------------------------------------------------------------
# Top, oyunculardan farklı olarak DeepSORT'un appearance-embedding mantığıyla
# takip edilmez (küçük, tekdüze görünümlü bir nesne için görünüm embedding'i
# güvenilir değildir). Bunun yerine "en yüksek güven skorlu tek top tespiti"
# + "kayıp karelerde son bilinen konumu kısa süre koru" mantığıyla basit bir
# tekil (singleton) takip yapıyoruz.
BALL_MAX_MISSING_FRAMES = 15  # top bu kadar karede tespit edilmezse "kayıp" sayılır

# ------------------------------------------------------------------
# VİDEO İŞLEME / ÇİZİM AYARLARI
# ------------------------------------------------------------------
BBOX_COLOR_DEFAULT = (0, 255, 0)   # BGR formatında yeşil (henüz takımı bilinmeyen nesneler için)
BBOX_COLOR_BALL = (0, 165, 255)    # BGR formatında turuncu (top için)
BBOX_THICKNESS = 2
ID_FONT_SCALE = 0.6
ID_FONT_THICKNESS = 2

# Çıkış videosunun codec'i. 'mp4v' çoğu platformda .mp4 için uyumludur.
VIDEO_CODEC = "mp4v"
