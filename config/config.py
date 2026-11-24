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

# ------------------------------------------------------------------
# VERİ DIŞA AKTARMA (ADIM 6) AYARLARI
# ------------------------------------------------------------------
EXPORT_CSV_FILENAME = "tracking_data.csv"              # ham, kare-kare takip verisi
EXPORT_SQL_FILENAME = "tracking_data.db"                # aynı veri, SQLite veritabanı olarak
EXPORT_SQL_TABLE_NAME = "tracking_data"
EXPORT_DISTANCE_SUMMARY_FILENAME = "player_distance_summary.csv"  # oyuncu başına özet
EXPORT_ANALYTICS_SUMMARY_FILENAME = "match_analytics_summary.json"
EXPORT_POSSESSION_FILENAME = "possession_timeline.csv"
EXPORT_HEATMAP_DIR = os.path.join(EXPORT_DIR, "heatmaps")
EXPORT_FLUSH_INTERVAL_FRAMES = 0                       # DataExporter varsayılanı (0 = bellekte tut; main.py CLI'da 2500 kullanır)

# Mesafe/hız hesabı (saha koordinatları üzerinden), sadece ARDIŞIK karelerde yapılır.
# Aradan kare kaçarsa (ör. oyuncu geçici olarak kayboldu) o segment mesafeye
# KATILMAZ -- aksi halde "ışınlanma" gibi görünen sahte, çok büyük mesafeler hesaplanır.
DISTANCE_CALC_MAX_FRAME_GAP = 1
# Kalman filtresinin TAHMİN ettiği (gerçek tespit olmayan) konumlar mesafe
# hesabına katılmaz -- tahmini konumlar gerçek hareketi yansıtmayabilir.
DISTANCE_CALC_EXCLUDE_PREDICTED = True
# Saha dışı (on_pitch=False) sayılan noktalar da mesafeye katılmaz (kalibrasyon
# hatası ya da kenar çizgisi dışına taşan tespitlerin mesafeyi şişirmesini önler).

# Mesafe ve hız hesaplamasında bbox titremesini azaltmak için yumuşatma ve fiziksel sınırlar
SPEED_SMOOTHING_WINDOW_FRAMES = 5
MAX_REALISTIC_PLAYER_SPEED_KMH = 38.0

# FIFA hız bantları (km/saat) ve sprint süresi eşiği
SPEED_ZONE_WALKING_MAX_KMH = 7.0
SPEED_ZONE_JOGGING_MAX_KMH = 14.0
SPEED_ZONE_RUNNING_MAX_KMH = 20.0
SPEED_ZONE_HSR_MAX_KMH = 25.0
SPRINT_MIN_DURATION_SEC = 0.5

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
# ayrımını biz 4. adımda (K-Means forma renk kümeleme) yapıyoruz.
#
# --- FİNE-TUNE EDİLMİŞ (FUTBOLA ÖZEL) MODEL KULLANMAK İSTERSEN ---
# Roboflow Universe'teki halka açık "football-players-detection" veri seti
# (bkz. https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
# ile fine-tune edilmiş bir YOLO modeli kullanabilirsin. Bu veri setinin sınıf
# sırası: 0=ball, 1=goalkeeper, 2=player, 3=referee. Kendi modelini eğitmek
# için utils/train_custom_yolo.py betiğini KENDİ GPU'lu ortamında (Google
# Colab, kendi makinen vb.) çalıştırabilirsin -- bu betik ve fine-tune süreci
# hakkında ayrıntılı açıklama o dosyanın başında.
#
# Fine-tune edilmiş modele geçmek için AŞAĞIDAKİ İKİ DEĞERİ DEĞİŞTİRMEN YETERLİ,
# başka hiçbir modüle (tracker.py, team_classifier.py, main.py) DOKUNMANA
# gerek yok -- bu dosyanın altındaki HUMAN_CLASS_NAMES, BALL_CLASS_NAME ve
# USES_FINE_TUNED_FOOTBALL_MODEL değerleri, TARGET_CLASS_IDS'ten OTOMATİK
# olarak türetiliyor ve tüm pipeline buna göre kendini ayarlıyor:
#
#   YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "football_yolov8_best.pt")
#   TARGET_CLASS_IDS = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
#
# (Farklı bir Roboflow projesi/veri seti kullanıyorsan, indirdiğin data.yaml
# dosyasındaki "names:" listesinin SIRASINI birebir buraya kopyala -- sıra
# yanlışsa sınıf ID'leri yanlış isimlere eşlenir.)
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "football_yolo11s_best.pt")  # M1 futbola özel fine-tune model
MODEL_DOWNLOAD_URL = ""  # Opsiyonel ağırlık indirme adresi (örn. GitHub Releases)

# Saha anahtar nokta (32 keypoint) YOLO-Pose kalibrasyon modeli (Phase A2)
PITCH_KEYPOINT_MODEL_PATH = os.path.join(MODELS_DIR, "football_pitch_pose.pt")
PITCH_KEYPOINT_MODEL_DOWNLOAD_URL = ""  # Opsiyonel ağırlık indirme adresi

# YOLO çıkarım (inference) çözünürlüğü, FP16 ve batch ayarları
DETECTION_IMG_SIZE = 960
DETECTION_HALF = True
DETECTION_BATCH_SIZE = 8

# Tespit güven eşiği: Bu değerin altındaki tespitler göz ardı edilir.
DETECTION_CONFIDENCE_THRESHOLD = 0.3

# IOU eşiği (Non-Max Suppression için): Örtüşen kutuların birleştirilme oranı.
DETECTION_IOU_THRESHOLD = 0.45

# Futbola özel fine-tune edilmiş 4 sınıflı model ID eşlemesi (data.yaml sırası):
#   0 -> ball
#   1 -> goalkeeper
#   2 -> player
#   3 -> referee
TARGET_CLASS_IDS = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}

# --- AŞAĞIDAKİLER TARGET_CLASS_IDS'TEN OTOMATİK TÜRETİLİR, ELLE DEĞİŞTİRME ---
#
# Top olarak sayılan sınıf ismi (varsayım: veri setinde her zaman "ball" olarak
# adlandırılmıştır -- farklıysa TARGET_CLASS_IDS'teki ismi güncelle, burası
# otomatik uyum sağlar).
BALL_CLASS_NAME = "ball"

# "İnsan" (oyuncu/kaleci/hakem, her ne olursa olsun top DIŞINDAKİ her şey)
# sayılan tüm sınıf isimleri. tracker.py'deki PlayerTracker, DeepSORT'a
# BUNLARI besler -- COCO modelinde bu sadece {"person"}, fine-tune futbol
# modelinde ise {"player", "goalkeeper", "referee"} olur.
HUMAN_CLASS_NAMES = {name for name in TARGET_CLASS_IDS.values() if name != BALL_CLASS_NAME}

# Model, hakemi YOLO SEVİYESİNDE ayrı bir sınıf olarak zaten tanıyor mu?
# (COCO modelinde HAYIR -- hakem/oyuncu ayrımını K-Means renk kümelemesiyle
# TAHMİN ETMEMİZ gerekiyor. Fine-tune futbol modelinde EVET -- bu durumda
# team_classifier.py, hakemleri renk kümelemesine hiç sokmadan doğrudan
# "Referee" etiketler; bu hem daha doğru hem de K-Means'in geriye kalan
# kümelemeyi SADECE Takım A/Takım B arasında yapmasını sağlar.)
USES_FINE_TUNED_FOOTBALL_MODEL = "referee" in HUMAN_CLASS_NAMES

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

# GPU varsa embedder'ı hızlandırmak için True yap (CUDA yoksa PlayerTracker otomatik CPU'ya düşer).
TRACKER_EMBEDDER_GPU = True
TRACKER_EMBEDDER_HALF = True

# 90 dakikalık maç ölçeklenebilirliği için takipçi tipi ("deepsort" veya "bytetrack")
# ve DeepSORT görünüm embedding hesaplama adımı (1 = her kare, >1 = ara karelerde Kalman+IoU)
TRACKER_TYPE = "deepsort"
TRACKER_EMBEDDING_STRIDE = 1
TRACKER_MAX_IOU_DISTANCE = 0.7
TRACKER_HIGH_CONF_THRESH = 0.45
TRACKER_LOW_CONF_THRESH = 0.05
TRACKER_NEW_TRACK_CONF_THRESH = 0.30

# ------------------------------------------------------------------
# TOP TAKİBİ (BALL TRACKING) AYARLARI
# ------------------------------------------------------------------
# Top, oyunculardan farklı olarak DeepSORT'un appearance-embedding mantığıyla
# takip edilmez (küçük, tekdüze görünümlü bir nesne için görünüm embedding'i
# güvenilir değildir). Bunun yerine "en yüksek güven skorlu tek top tespiti"
# + "kayıp karelerde sönümlenen hızla konum tahmini" mantığıyla tekil takip yapıyoruz.
BALL_MAX_MISSING_FRAMES = 15  # top bu kadar karede tespit edilmezse "kayıp" sayılır
BALL_VELOCITY_DAMPING = 0.85
BALL_MAX_VELOCITY_PX_PER_FRAME = 150.0
BALL_INTERPOLATE_MAX_GAP = 15

# ------------------------------------------------------------------
# TAKIM SINIFLANDIRMA (K-MEANS) AYARLARI
# ------------------------------------------------------------------
# Bazı görüntülerde (amatör maç, antrenman klibi, drone/stok footage, hakemin
# sürekli kadraj dışında olduğu kısa klipler vb.) SAHADA GÖRÜNÜR bir hakem
# olmayabilir ya da forma rengi oyunculardan yeterince AYIRT EDİCİ olmayabilir.
# Bu durumda K-Means'in "en az üyeli küme = hakem" varsayımı YANLIŞ POZİTİF
# üretir -- gerçekte hakem OLMAYAN bir grup oyuncuyu "Hakem" diye etiketler
# (bu tam olarak stok/amatör görüntülerde sık karşılaşılan bir durumdur).
#
# Görüntünde net, oyunculardan farklı renkli forma giyen bir hakem YOKSA
# ya da "Hakem" etiketinin yanlışlıkla oyunculara yapıştığını görüyorsan,
# bunu False yap -- K-Means artık SADECE 2 küme (Takım A / Takım B) kullanır,
# hiçbir grup zorla "Hakem" etiketlenmez.
TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT = True

# (Sadece TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT=True iken kullanılır.)
# En küçük kümenin GERÇEKTEN bir azınlık (yani muhtemelen hakem) sayılabilmesi
# için, boyutunun diğer iki kümenin ortalamasının bu ORANDAN küçük olması
# gerekir. Örn. 0.6 demek: en küçük küme, diğerlerinin ortalamasının
# %60'ından DAHA BÜYÜKSE, muhtemelen gerçek bir hakem grubu DEĞİLDİR --
# sadece K-Means'in kusurlu bir renk ayrımıdır. Bu durumda konsola uyarı
# basılır (ama işlem yine de devam eder, en küçük küme yine "Hakem" sayılır --
# kalıcı çözüm için yukarıdaki bayrağı False yapmak gerekir).
TEAM_CLASSIFIER_REFEREE_SIZE_RATIO_THRESHOLD = 0.6

# Küme sayısı, iki farklı sebepten 2'ye düşebilir:
#   1) Fine-tune futbol modeli kullanılıyorsa (hakem zaten YOLO'dan ayrı sınıf)
#   2) TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT=False ise (görüntüde ayırt
#      edilebilir bir hakem yok/olmadığı varsayılıyor)
TEAM_CLASSIFIER_N_CLUSTERS = (
    2 if USES_FINE_TUNED_FOOTBALL_MODEL
    else (3 if TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT else 2)
)

# Renk uzayı ("LAB" ışık/gölge değişimlerine daha dayanıklıdır, "BGR" klasik mod)
TEAM_COLOR_SPACE = "LAB"
# Fit sonrası yeni gelen bir track_id'nin takımını kalıcı olarak önbelleğe
# kilitlemeden önce görülmesi gereken minimum örnek sayısı
TEAM_CLASSIFIER_MIN_SAMPLES_FOR_LOCK = 3
# Kilitli bir track_id için periyodik forma rengi yenileme aralığı (kare/gözlem)
TEAM_CLASSIFIER_LOCK_REFRESH_INTERVAL = 5
# Hareketli çoğunluk oylaması pencere boyutu ve kilit güncelleme eşiği
TEAM_CLASSIFIER_VOTE_WINDOW = 7
TEAM_CLASSIFIER_VOTE_MIN_AGREEMENT = 0.6
# İlk fit sonrası sınırlı K-Means küme merkezi iyileştirme sayısı ve bellek limitleri
TEAM_CLASSIFIER_MAX_REFITS = 3
TEAM_CLASSIFIER_MAX_TRACKED_VOTES = 2048
TEAM_CLASSIFIER_MAX_FEATURE_BANK = 512
# Kaleci -> takım mekânsal eşleştirmesi ve kilitleme eşikleri
TEAM_CLASSIFIER_GK_MIN_SEPARATION_PX = 40.0
TEAM_CLASSIFIER_GK_MIN_SEPARATION_M = 5.0  # kalibrasyon varken aynı kontrol saha metresiyle
TEAM_CLASSIFIER_GK_X_EMA_ALPHA = 0.15
TEAM_CLASSIFIER_GK_LOCK_MIN_VOTES = 25
TEAM_CLASSIFIER_GK_LOCK_MIN_AGREEMENT = 0.8
TEAM_CLASSIFIER_GK_VOTE_WINDOW = 40
# Video bitiminde K-Means fit edilmeden önceki ilk karelerdeki None takım
# etiketlerini sonradan öğrenilen track_id -> team eşlemesiyle doldur
TEAM_CLASSIFIER_BACKFILL_TEAMS = True

# K-Means'i "fit" etmeden (eğitmeden) önce en az kaç FARKLI oyuncunun (track_id)
# forma rengi örneği toplanmalı? Çok erken fit edersek (örn. sadece 2-3 oyuncu
# görülmüşken), kümeler yanlış oturur ve tüm maç boyunca yanlış takım
# etiketleri üretilebilir. Genelde ilk birkaç saniyede her iki takımdan ve
# hakemden en az birkaç örnek görülmesi yeterlidir.
TEAM_CLASSIFIER_MIN_UNIQUE_TRACKS = 10

# Her track_id için bellekte tutulacak maksimum forma rengi örneği sayısı.
# Bu örneklerin ortalaması, o oyuncunun "referans rengi" olarak kullanılır.
# Çok büyük tutmak gereksiz bellek kullanır, çok küçük tutmak tek bir kötü
# kareden etkilenmeyi artırır.
TEAM_COLOR_HISTORY_MAX_SAMPLES = 30
# Henüz kilitlenmemiş (unlocked) gürültü track'leri için bellek üst sınırı
TEAM_COLOR_HISTORY_MAX_UNLOCKED_TRACKS = 512

# Forma rengi çıkarılırken bounding box'ın hangi dikey aralığı kullanılsın?
# (0.0 = kutunun tepesi, 1.0 = kutunun tabanı)
# Baştaki saçı/kafayı ve alttaki şortu/çorabı dışarıda bırakıp sadece
# formanın olduğu gövde bölgesini almak için varsayılan olarak 0.15-0.55 arası.
JERSEY_CROP_TOP_RATIO = 0.15
JERSEY_CROP_BOTTOM_RATIO = 0.55

# Yatayda da kenarlardaki komşu oyuncu/arka plan sızıntısını azaltmak için
# kutunun sadece orta genişliğini kullanıyoruz.
JERSEY_CROP_LEFT_RATIO = 0.2
JERSEY_CROP_RIGHT_RATIO = 0.8

# ------------------------------------------------------------------
# KAMERA HAREKETİ TAKİBİ (OPTICAL FLOW) AYARLARI
# ------------------------------------------------------------------
# Yayın kamerası maç sırasında pan/tilt/zoom yaptığında, tek bir statik
# homografi matrisi geçersizleşir. Bu ayarlar, arka plandaki sabit noktaları
# (çizgiler, reklam panoları vb.) kare-kare takip ederek kamera hareketini
# tahmin etmemizi ve homografiyi buna göre güncellememizi sağlar.

# goodFeaturesToTrack: takip edilecek "köşe" (corner) noktalarını bulan fonksiyon.
CAMERA_MOTION_MAX_CORNERS = 300       # aranacak maksimum özellik noktası sayısı
CAMERA_MOTION_QUALITY_LEVEL = 0.01    # köşe kalitesi eşiği (düşük = daha fazla nokta, daha gürültülü)
CAMERA_MOTION_MIN_DISTANCE = 15       # iki nokta arası minimum piksel mesafesi

# Oyuncu/top bounding box'larının etrafına eklenen güvenlik payı (piksel).
# Bu bölgeler "hareketli nesne" sayılıp özellik noktası aranmaz -- sadece
# SABİT arka plan (çim, çizgiler, reklam panosu) üzerinde nokta ararız.
PLAYER_MASK_PADDING_PX = 10

# Lucas-Kanade Optical Flow parametreleri
OPTICAL_FLOW_WIN_SIZE = (21, 21)      # her noktanın izlendiği pencere boyutu
OPTICAL_FLOW_MAX_LEVEL = 3            # piramit seviyesi (büyük hareketleri yakalamak için)
OPTICAL_FLOW_CRITERIA_COUNT = 30      # maksimum iterasyon
OPTICAL_FLOW_CRITERIA_EPS = 0.01      # yakınsama hassasiyeti

# Bir karede güvenilir bir hareket tahmini yapabilmek için gereken minimum
# eşleşmiş (başarıyla takip edilmiş) nokta sayısı. Bunun altına düşülürse
# (örn. kamera çok hızlı hareket etti, motion blur oluştu) o karede hareket
# GÜNCELLENMEZ, son bilinen matris korunur.
CAMERA_MOTION_MIN_TRACKED_POINTS = 15

# Kare-kare hareket homografisi hesaplanırken RANSAC eşiği (piksel).
CAMERA_MOTION_RANSAC_THRESHOLD = 3.0

# SAHNE KESMESİ (SCENE CUT) TESPİTİ
# Yayın başka bir kameraya (tekrar izleme, close-up vb.) aniden geçtiğinde,
# iki ardışık karenin histogramı çok farklılaşır. Bu korelasyon değerinin
# ALTINA düşülürse "sahne kesmesi" kabul edilir ve kamera takibi sıfırlanır
# (o kareden itibaren saha koordinatı üretilmez, yeniden kalibrasyon gerekir).
SCENE_CUT_CORRELATION_THRESHOLD = 0.5
SCENE_CUT_HSV_CORRELATION_THRESHOLD = 0.45
SCENE_CUT_FADE_WINDOW = 8
SCENE_CUT_FADE_CORRELATION_THRESHOLD = 0.58
CAMERA_MOTION_SCALE = 0.5
CAMERA_MOTION_SCALE_MIN_WIDTH = 960
CAMERA_ORB_MAX_FEATURES = 600
CAMERA_KEYFRAME_MAX_BANK = 8
CAMERA_KEYFRAME_MIN_INTERVAL = 25
CAMERA_RECOVERY_MIN_INLIERS = 14
CAMERA_RECOVERY_MIN_INLIER_RATIO = 0.25
CAMERA_DRIFT_CHECK_INTERVAL = 45
CAMERA_MAIN_GREEN_MIN_RATIO = 0.22

# ------------------------------------------------------------------
# PERSPEKTİF DÖNÜŞÜMÜ (HOMOGRAPHY) AYARLARI
# ------------------------------------------------------------------
# STANDART FUTBOL SAHASI BOYUTLARI (FIFA/IFAB önerisi, metre cinsinden).
# Koordinat sistemi: orijin (0,0) sol üst köşe, x ekseni uzun kenar (uzunluk),
# y ekseni kısa kenar (genişlik) boyunca artar.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
CENTER_CIRCLE_RADIUS_M = 9.15  # FIFA/IFAB standardı

# Sahadaki, kamerada genelde net görülen sabit referans noktalarının GERÇEK
# metre cinsinden konumları. Bunlar değişmez -- her sahada aynıdır (standart
# ölçülere göre). Homografi hesaplamak için bu noktalardan en az 4 tanesinin
# YAYIN KAMERASINDAKİ piksel karşılığını (SOURCE_CALIBRATION_POINTS) bulmamız
# gerekiyor.
#
# NOT -- orta yuvarlağın 4 kritik noktası (üst/alt/sol/sağ):
# Standart bir yayın kamerası genelde sahanın SADECE ORTA KISMINI net gösterir
# -- köşe bayrakları, ceza sahaları çoğu zaman kadraj dışında kalır. Orta
# yuvarlağın orta çizgiyle kesiştiği 4 nokta ise (yarıçap = 9.15m) HEMEN HEMEN
# HER kamera açısında net görünür. Bu yüzden bunlar, köşe/ceza sahası
# noktalarından daha PRATİK ve GÜVENİLİR bir kalibrasyon seçimi olabilir.
PITCH_LANDMARKS_METERS = {
    "top_left_corner": (0.0, 0.0),
    "top_right_corner": (105.0, 0.0),
    "bottom_left_corner": (0.0, 68.0),
    "bottom_right_corner": (105.0, 68.0),
    "halfway_top": (52.5, 0.0),
    "halfway_bottom": (52.5, 68.0),
    "center_spot": (52.5, 34.0),
    "center_circle_top": (52.5, 34.0 - CENTER_CIRCLE_RADIUS_M),      # (52.5, 24.85)
    "center_circle_bottom": (52.5, 34.0 + CENTER_CIRCLE_RADIUS_M),   # (52.5, 43.15)
    "center_circle_left": (52.5 - CENTER_CIRCLE_RADIUS_M, 34.0),     # (43.35, 34.0)
    "center_circle_right": (52.5 + CENTER_CIRCLE_RADIUS_M, 34.0),    # (61.65, 34.0)
    "left_penalty_spot": (11.0, 34.0),
    "right_penalty_spot": (94.0, 34.0),
    "left_penalty_box_top_left": (0.0, 13.84),
    "left_penalty_box_top_right": (16.5, 13.84),
    "left_penalty_box_bottom_left": (0.0, 54.16),
    "left_penalty_box_bottom_right": (16.5, 54.16),
    "right_penalty_box_top_left": (88.5, 13.84),
    "right_penalty_box_top_right": (105.0, 13.84),
    "right_penalty_box_bottom_left": (88.5, 54.16),
    "right_penalty_box_bottom_right": (105.0, 54.16),
    "left_six_yard_box_top_left": (0.0, 24.84),
    "left_six_yard_box_top_right": (5.5, 24.84),
    "left_six_yard_box_bottom_left": (0.0, 43.16),
    "left_six_yard_box_bottom_right": (5.5, 43.16),
    "right_six_yard_box_top_left": (99.5, 24.84),
    "right_six_yard_box_top_right": (105.0, 24.84),
    "right_six_yard_box_bottom_left": (99.5, 43.16),
    "right_six_yard_box_bottom_right": (105.0, 43.16),
}

# !!! BURASI HER VİDEO/KAMERA AÇISI İÇİN ELLE DOLDURULMALI !!!
# Yukarıdaki isimlerden seçtiğin en az 4 tanesinin, senin video görüntündeki
# PİKSEL koordinatlarını buraya gir. utils/calibrate_homography.py aracı,
# --frame-index (ya da --timestamp) ile seçtiğin karedeki tıkladığın
# noktaların piksel koordinatlarını ekrana basar -- oradan kopyalayıp buraya
# yapıştırman yeterli.
# Format: {"landmark_ismi": (piksel_x, piksel_y), ...}
# Örnek (kendi videonla değiştirmen gerekir):
SOURCE_CALIBRATION_POINTS = {
    # "top_left_corner": (145, 210),
    # "top_right_corner": (1780, 195),
    # "left_penalty_box_top_left": (145, 340),
    # "left_penalty_box_bottom_left": (160, 610),
}

# --- HANGİ KAREDE KALİBRASYON YAPILDI? ---
# YUKARIDAKİ SOURCE_CALIBRATION_POINTS'İ HANGİ KAREYE BAKARAK DOLDURDUYSAN,
# o karenin numarasını (frame_idx) buraya yaz. Videonun İLK karesinde
# kalibrasyon yapmak ZORUNDA DEĞİLSİN -- video başında saha net görünmüyorsa
# (intro, tekrar izleme, farklı bir açı vb.) sahanın net göründüğü herhangi
# bir kareyi seçebilirsin (bkz. utils/calibrate_homography.py --frame-index).
#
# ÖNEMLİ: Bu değer, kamera hareketi telafisinin (camera_motion_tracker.py)
# "referans karesi" ile BİREBİR AYNI OLMAK ZORUNDADIR -- main.py bunu otomatik
# ayarlar (bkz. main.py'deki ilgili yorum), sen sadece bu sayıyı doğru girmen
# yeterli. Bu kareden ÖNCEKİ kareler için saha (metre) koordinatı üretilmez
# (pitch_x/y = None kalır) -- kamera hareketi takibi henüz "referans" almadığı
# için o kareler için güvenilir bir dönüşüm yoktur.
CALIBRATION_FRAME_INDEX = 0

# findHomography için RANSAC eşiği (piksel). Kalibrasyon noktalarında küçük
# tıklama hataları olabileceğinden RANSAC, aykırı (outlier) noktaları eleyerek
# daha sağlam bir dönüşüm matrisi bulmaya yardımcı olur. 4 nokta ile tam
# çözüm (getPerspectiveTransform) kullanılır; 4'ten fazla nokta girilirse
# RANSAC devreye girer.
HOMOGRAPHY_RANSAC_THRESHOLD = 5.0
HOMOGRAPHY_MIN_ABS_DET = 1e-10
HOMOGRAPHY_MAX_CONDITION_NUMBER = 1e8
HOMOGRAPHY_MIN_HOMOGENEOUS_W = 1e-6

# Otomatik saha kalibrasyonu (Phase A2: 32 keypoint YOLO-Pose + hibrit takip)
AUTO_CALIBRATE_PITCH = True
PITCH_CALIBRATION_REFRESH_INTERVAL = 15
PITCH_CALIBRATION_SMOOTH_FRAMES = 5
PITCH_KEYPOINT_CONF_THRESHOLD = 0.50
PITCH_KEYPOINT_DET_CONF_THRESHOLD = 0.25
PITCH_KEYPOINT_IMGSZ = 640
PITCH_KEYPOINT_HALF = True
PITCH_CALIBRATION_MIN_KEYPOINTS = 4
PITCH_CALIBRATION_RANSAC_PX = 8.0
PITCH_CALIBRATION_MAX_REPROJ_ERR_PX = 12.0
PITCH_CALIBRATION_MAX_REPROJ_ERR_M = 2.0

# A2: Saha çizgisine oturtma — model tablosunu beyaz çizgilere hizalayan ince ayar (modules/pitch_line_refiner.py)
PITCH_LINE_REFINE_ENABLED = True
PITCH_LINE_REFINE_LOOSE_RANSAC_PX = 25.0     # ek başlangıç adayı: gevşek RANSAC (model noktaları 20-50 px sapabiliyor)
PITCH_LINE_REFINE_WORK_HEIGHT = 540          # oturtma bu yükseklikteki küçültülmüş karede yapılır (hız)
PITCH_LINE_REFINE_SAMPLE_STEP_M = 1.0        # saha çiziminden örnek aralığı (metre)
PITCH_LINE_REFINE_TOPHAT_THRESHOLD = 18      # çizgi maskesi parlaklık eşiği (top-hat)
PITCH_LINE_REFINE_INLIER_PX = 3.0            # çizgiye "oturdu" sayılma mesafesi (1080p pikseli)
PITCH_LINE_REFINE_MIN_INLIER_RATIO = 0.50    # kabul için çizgiye oturan örnek oranı alt sınırı (yanlış tablolar <= 0,47 ölçüldü)
PITCH_LINE_REFINE_MAX_SHIFT_FRAC = 0.10      # örneklerin ortanca yer değiştirmesi <= kare köşegeni x bu oran

# A2-3: Model–optical flow uyuşmazlığına göre kademeli kabul (metre, 3x3 referans ızgarasının ortancası)
PITCH_CAL_DEADBAND_M = 0.5          # bu farkın altında tablo değiştirilmez (gürültü titremesini önler)
PITCH_CAL_SOFT_M = 1.0              # bu farka kadar yumuşak geçişle hemen kabul
PITCH_CAL_HARD_M = 3.0              # bunun üstü doğrulandıktan sonra sert geçiş + calib_reset
PITCH_CAL_CONFIRM_REFRESHES = 2     # SOFT üstü farkta kabul için gereken ek tutarlı yenileme sayısı
PITCH_CAL_BLEND_MAX_STEP_M = 0.05   # yumuşak geçişte referans noktaların kare başına en fazla kayması

# A2-5: Görüntü profili ("auto" | "fixed" | "broadcast")
FOOTAGE_PROFILE = "auto"
PROFILE_DETECT_SEC = 5.0                    # otomatik algılama penceresi
PROFILE_FIXED_MAX_PAN_RATIO = 0.05          # pencerede görüntü merkezinin toplam yolu / kare genişliği
PROFILE_FIXED_MAX_SCALE_CHANGE = 0.03       # pencerede en büyük ölçek (zoom) değişimi
PITCH_CAL_FIXED_CANDIDATES = 10             # sabit kamerada kilit için toplanan en fazla aday tablo
PITCH_CAL_FIXED_HEALTH_SEC = 10.0           # sabit kamerada sağlık kontrolü aralığı

# A2-2: Kare bazında homografi kaydı
EXPORT_HOMOGRAPHY_LOG_FILENAME = "homography_log.csv"
EXPORT_CALIBRATION_SUMMARY_FILENAME = "calibration_summary.json"

# Roboflow football-field-detection 32-keypoint köşe sırası (0..31),
# 105.0m x 68.0m IFAB standart saha koordinatlarına hizalı.
PITCH_KEYPOINT_NAMES_32 = (
    "top_left_corner",                      # 0  (#1)
    "left_penalty_box_top_left",            # 1  (#2)
    "left_six_yard_box_top_left",           # 2  (#3)
    "left_six_yard_box_bottom_left",        # 3  (#4)
    "left_penalty_box_bottom_left",         # 4  (#5)
    "bottom_left_corner",                   # 5  (#6)
    "left_six_yard_box_top_right",          # 6  (#7)
    "left_six_yard_box_bottom_right",       # 7  (#8)
    "left_penalty_spot",                    # 8  (#9)
    "left_penalty_box_top_right",           # 9  (#10)
    "left_penalty_arc_top",                 # 10 (#11) ceza yayı x ceza sahası çizgisi
    "left_penalty_arc_bottom",              # 11 (#12)
    "left_penalty_box_bottom_right",        # 12 (#13)
    "halfway_top",                          # 13 (#14)
    "center_circle_top",                    # 14 (#15)
    "center_circle_bottom",                 # 15 (#16)
    "halfway_bottom",                       # 16 (#17)
    "right_penalty_box_top_left",           # 17 (#18)
    "right_penalty_arc_top",                # 18 (#19) ceza yayı x ceza sahası çizgisi
    "right_penalty_arc_bottom",             # 19 (#20)
    "right_penalty_box_bottom_left",        # 20 (#21)
    "right_penalty_spot",                   # 21 (#22)
    "right_six_yard_box_top_left",          # 22 (#23)
    "right_six_yard_box_bottom_left",       # 23 (#24)
    "top_right_corner",                     # 24 (#25)
    "right_penalty_box_top_right",          # 25 (#26)
    "right_six_yard_box_top_right",         # 26 (#27)
    "right_six_yard_box_bottom_right",      # 27 (#28)
    "right_penalty_box_bottom_right",       # 28 (#29)
    "bottom_right_corner",                  # 29 (#30)
    "center_circle_left",                   # 30 (#31)
    "center_circle_right",                  # 31 (#32)
)

# 10/11/18/19: Roboflow etiketleri ceza yayının ceza sahası çizgisini kestiği noktadadır
# (y = 34 ± sqrt(9.15² - 5.5²) = 34 ± 7.3125), kale alanı hizasında (24.84/43.16) DEĞİL.
# Veri setinden ölçüldü (A2-7 nokta sırası kontrolü, ~115 etiket/nokta, ~1.85 m sistematik fark).
_PENALTY_ARC_HALF_CHORD_M = (CENTER_CIRCLE_RADIUS_M ** 2 - 5.5 ** 2) ** 0.5  # 7.3125

PITCH_KEYPOINTS_32_METERS = (
    (0.0, 0.0),                                         # 0: top_left_corner
    (0.0, 13.84),                                       # 1: left_penalty_box_top_left
    (0.0, 24.84),                                       # 2: left_six_yard_box_top_left
    (0.0, 43.16),                                       # 3: left_six_yard_box_bottom_left
    (0.0, 54.16),                                       # 4: left_penalty_box_bottom_left
    (0.0, 68.0),                                        # 5: bottom_left_corner
    (5.5, 24.84),                                       # 6: left_six_yard_box_top_right
    (5.5, 43.16),                                       # 7: left_six_yard_box_bottom_right
    (11.0, 34.0),                                       # 8: left_penalty_spot
    (16.5, 13.84),                                      # 9: left_penalty_box_top_right
    (16.5, 34.0 - _PENALTY_ARC_HALF_CHORD_M),           # 10: left_penalty_arc_top
    (16.5, 34.0 + _PENALTY_ARC_HALF_CHORD_M),           # 11: left_penalty_arc_bottom
    (16.5, 54.16),                                      # 12: left_penalty_box_bottom_right
    (52.5, 0.0),                                        # 13: halfway_top
    (52.5, 34.0 - CENTER_CIRCLE_RADIUS_M),              # 14: center_circle_top (52.5, 24.85)
    (52.5, 34.0 + CENTER_CIRCLE_RADIUS_M),              # 15: center_circle_bottom (52.5, 43.15)
    (52.5, 68.0),                                       # 16: halfway_bottom
    (88.5, 13.84),                                      # 17: right_penalty_box_top_left
    (88.5, 34.0 - _PENALTY_ARC_HALF_CHORD_M),           # 18: right_penalty_arc_top
    (88.5, 34.0 + _PENALTY_ARC_HALF_CHORD_M),           # 19: right_penalty_arc_bottom
    (88.5, 54.16),                                      # 20: right_penalty_box_bottom_left
    (94.0, 34.0),                                       # 21: right_penalty_spot
    (99.5, 24.84),                                      # 22: right_six_yard_box_top_left
    (99.5, 43.16),                                      # 23: right_six_yard_box_bottom_left
    (105.0, 0.0),                                       # 24: top_right_corner
    (105.0, 13.84),                                     # 25: right_penalty_box_top_right
    (105.0, 24.84),                                     # 26: right_six_yard_box_top_right
    (105.0, 43.16),                                     # 27: right_six_yard_box_bottom_right
    (105.0, 54.16),                                     # 28: right_penalty_box_bottom_right
    (105.0, 68.0),                                      # 29: bottom_right_corner
    (52.5 - CENTER_CIRCLE_RADIUS_M, 34.0),              # 30: center_circle_left (43.35, 34.0)
    (52.5 + CENTER_CIRCLE_RADIUS_M, 34.0),              # 31: center_circle_right (61.65, 34.0)
)

# Kuş bakışı 2D saha görselleştirmesi (minimap) boyutları (piksel).
# Saha oranını (105:68) koruyacak şekilde otomatik hesaplanır.
PITCH_MINIMAP_WIDTH_PX = 600
PITCH_MINIMAP_HEIGHT_PX = int(PITCH_MINIMAP_WIDTH_PX * (PITCH_WIDTH_M / PITCH_LENGTH_M))
PITCH_MINIMAP_MARGIN_PX = 20

# ------------------------------------------------------------------
# TAKTİK GÖRÜNÜM (MİNİMAP) -- ÇIKTI VİDEOSUNA GÖMME AYARLARI
# ------------------------------------------------------------------
# NOT: draw_pitch_minimap() daha önce main.py'ye hiç bağlanmamıştı -- sadece
# perspective_transformer.py'nin kendi test bloğunda kullanılıyordu. Şimdi
# main.py, kalibrasyon yapılmışsa (transformer.is_calibrated=True) her karede
# bu minimap'i üretip video_processor.py'nin overlay_minimap() metoduyla
# çıktı videosunun bir köşesine gömüyor (resim-içinde-resim / "tactical cam" tarzı).
# Kalibrasyon yapılmamışsa minimap hiç üretilmez/gösterilmez (saha koordinatı
# olmadan anlamlı bir minimap çizilemez).
SHOW_TACTICAL_MINIMAP = True
MINIMAP_OVERLAY_SCALE = 0.3          # video genişliğinin bu oranı kadar küçültülüp yerleştirilir
MINIMAP_OVERLAY_MARGIN_PX = 20       # video kenarından boşluk (piksel)
MINIMAP_OVERLAY_POSITION = "bottom_right"  # "bottom_right" | "bottom_left" | "top_right" | "top_left"

# ------------------------------------------------------------------
# VİDEO İŞLEME / ÇİZİM AYARLARI
# ------------------------------------------------------------------
BBOX_COLOR_DEFAULT = (0, 255, 0)      # BGR formatında yeşil (henüz takımı bilinmeyen nesneler için)
BBOX_COLOR_BALL = (0, 165, 255)       # BGR formatında turuncu (top için)
BBOX_COLOR_TEAM_A = (255, 191, 0)     # canlı mavi
BBOX_COLOR_TEAM_B = (0, 0, 255)       # kırmızı
BBOX_COLOR_REFEREE = (0, 255, 255)    # sarı
BBOX_COLOR_GOALKEEPER = (255, 0, 255) # mor/magenta (kaleci için)
BBOX_THICKNESS = 2
ID_FONT_SCALE = 0.6
ID_FONT_THICKNESS = 2

# Çıkış videosunun codec'i. 'mp4v' çoğu platformda .mp4 için uyumludur.
VIDEO_CODEC = "mp4v"

# ------------------------------------------------------------------
# TAKTİK ANALİZ VE ISI HARİTASI (ADIM 7) AYARLARI
# ------------------------------------------------------------------
GENERATE_ANALYTICS_REPORT = True
ALLOW_PIXEL_FALLBACK_METRICS = False
POSSESSION_MAX_PLAYER_BALL_DIST_M = 2.5
POSSESSION_MAX_PLAYER_BALL_DIST_PX = 80.0
POSSESSION_SMOOTHING_FRAMES = 5
POSSESSION_MIN_CONTROL_SEC = 0.08
POSSESSION_MIN_CONTROL_FRAMES = 2
POSSESSION_MAX_BALL_CONTROL_SPEED_MPS = 15.0
BALL_AERIAL_DIRECTION_CHANGE_MIN_DEG = 25.0
BALL_AERIAL_DECEL_MIN_RATIO = 0.35
PASS_MIN_DURATION_SEC = 0.08
PASS_MAX_GAP_SEC = 4.0
PASS_MAX_SPEED_MPS = 42.0
PASS_MIN_BALL_MOVE_M = 2.0
PASS_MIN_BALL_MOVE_PX = 35.0
HALFTIME_MIN_GAP_SEC = 60.0
HALFTIME_MIN_FRAMES_FOR_SWAP = 60

# ------------------------------------------------------------------
# TAKTİK VE SCOUTING MAKİNE ÖĞRENMESİ MOTORU (ADIM 8) AYARLARI
# ------------------------------------------------------------------
GENERATE_ML_REPORT = True
EXPORT_ML_DIR = os.path.join(EXPORT_DIR, "ml_reports")
EXPORT_ML_SUMMARY_FILENAME = "ml_analysis_summary.json"
EXPORT_ML_REPORT_FILENAME = "ml_analysis_report.md"
EXPORT_ML_PLAYER_PROFILES_FILENAME = "ml_player_profiles.csv"
EXPORT_ML_TEMPO_WINDOWS_FILENAME = "ml_tempo_windows.csv"
EXPORT_ML_ANOMALIES_FILENAME = "ml_anomalies.csv"
EXPORT_ML_PASS_PREDICTIONS_FILENAME = "ml_pass_predictions.csv"

ML_RANDOM_STATE = 42
ML_TRAIN_RATIO = 0.60
ML_VAL_RATIO = 0.20
ML_TEST_RATIO = 0.20
ML_MIN_SUPERVISED_EVENTS = 30
ML_ALLOW_SYNTHETIC_FALLBACK = False
ML_SYNTHETIC_MATCH_EVENTS = 1500
ML_XT_GRID_COLS = 12
ML_XT_GRID_ROWS = 8
ML_XT_GRID_PATH = os.path.join(BASE_DIR, "data", "reference", "xt_grid_karun_singh_12x8.json")
ML_MIN_TRACK_FRAMES = 30
ML_CLUSTER_K_MIN = 2
ML_CLUSTER_K_MAX = 6
ML_TEMPO_WINDOW_SEC = 4.0
ML_TEMPO_MIN_SAMPLES_PER_WINDOW = 15
ML_TEMPO_DROP_Z_THRESHOLD = -1.5
ML_TEMPO_DROP_PCT_THRESHOLD = 0.25
ML_ANOMALY_CONTAMINATION = 0.08
ML_ANOMALY_MAD_Z_THRESHOLD = 3.5
ML_PIXEL_MAX_SPEED_PX_PER_SEC = 800.0

# ------------------------------------------------------------------
# GÜVENLİK VE ÜRETİM (PRODUCTION API) AYARLARI
# ------------------------------------------------------------------
# Yüklenen video dosyaları için kaynak tükenmesi (DoS) ve sahte dosya sınırları
SECURITY_MAX_VIDEO_SIZE_MB = 4096
SECURITY_MAX_VIDEO_WIDTH = 4096
SECURITY_MAX_VIDEO_HEIGHT = 2304
SECURITY_MAX_VIDEO_DURATION_SEC = 7500.0
SECURITY_MAX_VIDEO_FPS = 120.0
SECURITY_MIN_VIDEO_FPS = 1.0
SECURITY_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

# YOLO ağırlık (Pickle / RCE) güvenliği ve bütünlük doğrulaması
SECURITY_ENFORCE_MODEL_DIR_WHITELIST = True
SECURITY_MODEL_SHA256 = os.environ.get("FOOTBALL_MODEL_SHA256") or None
SECURITY_PITCH_MODEL_SHA256 = os.environ.get("FOOTBALL_PITCH_MODEL_SHA256") or None
SECURITY_ALLOWED_OFFICIAL_YOLO_BASENAMES = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
}

# API iş kuyruğu, hız sınırı ve kimlik doğrulama ayarları (gizli anahtarlar YALNIZCA env'den okunur)
API_JOBS_DIR = os.environ.get("FOOTBALL_JOBS_DIR") or os.path.join(BASE_DIR, "data", "jobs")
API_MAX_CONCURRENT_GPU_JOBS = 1
API_MAX_QUEUED_JOBS = 16
API_RATE_LIMIT_REQUESTS_PER_MIN = 30
API_UPLOAD_RATE_LIMIT_PER_MIN = 5
API_JWT_TTL_SECONDS = 3600
API_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FOOTBALL_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

