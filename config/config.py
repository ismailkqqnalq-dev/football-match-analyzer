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
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8x.pt")  # veya kendi fine-tune modelin

# YOLO çıkarım (inference) çözünürlüğü, FP16 ve batch ayarları
DETECTION_IMG_SIZE = 960
DETECTION_HALF = True
DETECTION_BATCH_SIZE = 8

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
POSSESSION_MAX_PLAYER_BALL_DIST_M = 2.5
POSSESSION_MAX_PLAYER_BALL_DIST_PX = 80.0
POSSESSION_SMOOTHING_FRAMES = 5
PASS_MAX_GAP_SEC = 4.0
PASS_MIN_BALL_MOVE_M = 2.0
PASS_MIN_BALL_MOVE_PX = 35.0
