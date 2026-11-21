"""
team_classifier.py
-------------------
ADIM 4: TAKIM SINIFLANDIRMASI (K-MEANS CLUSTERING)

Bu modülün görevi: her oyuncunun forma rengine bakarak onu Takım A, Takım B
veya Hakem olarak etiketlemek.

GENEL STRATEJİ (neden bu şekilde tasarlandı?)
    1) Her tespit edilen oyuncu kutusundan "forma rengini" çıkarmamız gerekiyor.
       Ama bir bounding box sadece formadan ibaret değildir -- içinde çim
       (arka plan), saç, ten rengi, şort da vardır. Bu yüzden:
         a) Önce kutunun sadece GÖVDE bölgesini (üst-orta kısmını) kırpıyoruz.
         b) Bu kırpılmış bölge içinde YİNE de arka plan (çim) sızıntısı olabilir,
            bu yüzden küçük bir 2-kümeli K-Means ile "forma pikselleri" ile
            "arka plan pikselleri"ni ayırıyoruz, arka planı köşe piksellerinden
            tespit edip atıyoruz. Geriye kalan küme = forma rengi.
    2) Tek bir karedeki tek bir oyuncunun rengine bakarak "Takım A mı Takım B mi"
       diye karar VERMİYORUZ -- çünkü ışık/gölge/açı o kareyi yanıltabilir.
       Bunun yerine her oyuncunun (track_id) birkaç karedeki renk örneklerini
       biriktirip ORTALAMASINI alıyoruz, böylece daha kararlı bir renk elde
       ediyoruz.
    3) Yeterli sayıda farklı oyuncudan (config.TEAM_CLASSIFIER_MIN_UNIQUE_TRACKS)
       renk örneği toplanınca, TÜM oyuncuların ortalama renkleri üzerinde
       TEK SEFERLİK bir K-Means(n_clusters=3) çalıştırıyoruz. Bu 3 küme bize
       "Takım A", "Takım B", "Hakem" gruplarını veriyor.
    4) Hangi kümenin "Hakem" olduğunu nasıl anlıyoruz?
       Sahada tipik olarak 3-4 hakem/yardımcı hakem, buna karşılık her takımdan
       ~11 oyuncu vardır. Yani HAKEM KÜMESİ, diğer iki kümeye göre BELİRGİN
       ŞEKİLDE DAHA AZ ÜYEYE sahiptir. Bu yüzden en az üyeli kümeyi "Hakem"
       olarak, kalan iki kümeyi de "Team A" / "Team B" olarak etiketliyoruz.
    5) K-Means bir kere fit edildikten sonra, ONU TEKRAR FIT ETMİYORUZ.
       Yeni gelen her oyuncu için sadece `kmeans.predict()` çağırıp hangi
       kümeye en yakın olduğunu buluyoruz ve SONUCU O TRACK_ID İÇİN
       ÖNBELLEKLİYORUZ (cache). Bu, aynı oyuncunun farklı karelerde
       "Takım A" iken sonra "Takım B" gibi görünüp titremesini (flickering)
       engelliyor -- ID sabitse takım etiketi de sabit kalır.

NOT -- FİNE-TUNE FUTBOL MODELİYLE ÇALIŞIRKEN (config.USES_FINE_TUNED_FOOTBALL_MODEL):
    Bazı YOLO modelleri (ör. Roboflow "football-players-detection") hakemi
    ZATEN kendi sınıfı olarak tanır ("referee"). Bu durumda hakemi renk
    kümelemesiyle TAHMİN ETMEMİZE gerek kalmaz -- onu doğrudan "Referee"
    etiketleriz ve renk geçmişine hiç eklemeyiz. Bu iki fayda sağlar:
        a) Hakem etiketi %100 doğru olur (K-Means'in "en az üyeli küme"
           sezgisine bağımlı kalmaz).
        b) K-Means artık SADECE Takım A/Takım B arasında ayrım yapar
           (config.TEAM_CLASSIFIER_N_CLUSTERS bu modda otomatik 2'ye düşer),
           bu da hakem formasının iki takımın renk kümesini bozmasını önler.
"""

import os
import sys
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import cv2
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class TeamClassifier:
    """
    Forma rengine göre oyuncuları Takım A / Takım B / Hakem olarak sınıflandıran sınıf.

    Kullanım örneği:
        classifier = TeamClassifier()
        for frame_idx, frame in vp.read_frames():
            ...
            tracked_players = classifier.assign_teams(tracked_players, frame)
            # artık her tracked_players[i]["team"] "Team A"/"Team B"/"Referee"/None olabilir
    """

    def __init__(
        self,
        n_clusters: int = config.TEAM_CLASSIFIER_N_CLUSTERS,
        min_unique_tracks: int = config.TEAM_CLASSIFIER_MIN_UNIQUE_TRACKS,
        max_history_per_track: int = config.TEAM_COLOR_HISTORY_MAX_SAMPLES,
        color_space: str = config.TEAM_COLOR_SPACE,
        min_samples_for_lock: int = config.TEAM_CLASSIFIER_MIN_SAMPLES_FOR_LOCK,
    ):
        self.n_clusters = n_clusters
        self.min_unique_tracks = min_unique_tracks
        self.max_history_per_track = max_history_per_track
        self.color_space = color_space
        self.min_samples_for_lock = max(1, min_samples_for_lock)

        # track_id -> [renk_örneği_1, renk_örneği_2, ...] (her biri 3 elemanlı renk vektörü)
        self.color_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # track_id -> "Team A" / "Team B" / "Referee" / "Goalkeeper" (kararlı önbellek)
        self.team_assignment: Dict[int, str] = {}

        # K-Means modelinin kendisi -- yeterli veri toplanınca bir kere fit edilir
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False

        # kmeans küme indeksi (0,1,2) -> "Team A"/"Team B"/"Referee" eşlemesi
        self.cluster_role_map: Dict[int, str] = {}

        print(
            f"[TeamClassifier] Oluşturuldu (color_space={self.color_space}). "
            f"En az {min_unique_tracks} farklı oyuncu görülünce K-Means kümeleme "
            f"otomatik olarak devreye girecek."
        )

    # ----------------------------------------------------------------
    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Bir oyuncunun bounding box'ından forma rengini (BGR) çıkarır.

        Adımlar:
          1) Kutunun sadece gövde bölgesini (baş ve bacaklar hariç) kırp.
          2) Bu bölgeyi küçük bir boyuta indir (hız için).
          3) 2 kümeli K-Means ile pikselleri "forma" ve "arka plan" olarak ayır.
          4) Köşe piksellerinin çoğunlukla hangi kümeye ait olduğuna bakarak
             o kümeyi "arka plan" (çim/tribün) olarak işaretle.
          5) Diğer kümenin ortalama rengini forma rengi olarak döndür.

        Dönüş
        -----
        Optional[np.ndarray]
            3 elemanlı BGR renk vektörü, ya da kutu çok küçük/geçersizse None.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Kutu koordinatlarını frame sınırları içine kelepçele (clip)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        box_height = y2 - y1
        box_width = x2 - x1

        # Çok küçük/geçersiz kutuları atla (ör. kısmen kadraj dışında kalan tespitler)
        if box_height < 10 or box_width < 10:
            return None

        # --- 1) Sadece gövde bölgesini kırp (config oranlarına göre) ---
        crop_y1 = y1 + int(box_height * config.JERSEY_CROP_TOP_RATIO)
        crop_y2 = y1 + int(box_height * config.JERSEY_CROP_BOTTOM_RATIO)
        crop_x1 = x1 + int(box_width * config.JERSEY_CROP_LEFT_RATIO)
        crop_x2 = x1 + int(box_width * config.JERSEY_CROP_RIGHT_RATIO)

        jersey_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if jersey_crop.size == 0:
            return None

        # --- 2) Hız için küçük sabit boyuta indir (renk kümelemede çözünürlük önemsiz) ---
        jersey_crop = cv2.resize(jersey_crop, (30, 30), interpolation=cv2.INTER_AREA)

        if self.color_space.upper() == "LAB":
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2LAB)

        # Pikselleri (N, 3) şekline getir -- her satır bir pikselin renk vektörü
        pixels = jersey_crop.reshape(-1, 3).astype(np.float32)

        # Tek renkli (uniform) kırpımlarda 2-kümeli K-Means çalıştırmaya gerek yok
        if len(np.unique(pixels, axis=0)) < 2:
            return pixels.mean(axis=0)

        # --- 3) 2 kümeli K-Means: forma pikselleri vs arka plan pikselleri ---
        # n_init=3 -> farklı başlangıç noktalarıyla 3 kez deneyip en iyisini seç
        # (küçük veri setinde ucuz bir işlem, kararlılığı artırır)
        pixel_kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
        pixel_labels = pixel_kmeans.fit_predict(pixels)

        # --- 4) Arka plan kümesini 4 köşedeki 3x3 blokların çoğunluk oyuyla tahmin et ---
        labels_2d = pixel_labels.reshape(30, 30)
        corner_blocks = np.concatenate([
            labels_2d[:3, :3].ravel(),
            labels_2d[:3, -3:].ravel(),
            labels_2d[-3:, :3].ravel(),
            labels_2d[-3:, -3:].ravel(),
        ])
        counts = np.bincount(corner_blocks, minlength=2)
        background_cluster = int(np.argmax(counts))
        jersey_cluster = 1 - background_cluster  # diğer küme = forma

        # --- 5) Forma kümesine ait piksellerin ortalama rengini hesapla ---
        jersey_pixels = pixels[pixel_labels == jersey_cluster]

        if len(jersey_pixels) == 0:
            # Nadir durum: tüm pikseller tek kümeye düştüyse, tüm crop'un ortalamasını al
            return pixels.mean(axis=0)

        return jersey_pixels.mean(axis=0)

    # ----------------------------------------------------------------
    def _collect_sample(self, track_id: int, color: np.ndarray):
        """Bir oyuncunun renk örneğini geçmişine ekler (maksimum limit ile)."""
        if self.is_fitted and track_id not in self.color_history:
            max_unlocked = int(getattr(config, "TEAM_COLOR_HISTORY_MAX_UNLOCKED_TRACKS", 512))
            if max_unlocked > 0:
                while len(self.color_history) >= max_unlocked:
                    oldest_tid = next(iter(self.color_history))
                    self.color_history.pop(oldest_tid, None)
        history = self.color_history[track_id]
        history.append(color)
        # Limit aşılırsa en eski örneği at (sliding window mantığı)
        if len(history) > self.max_history_per_track:
            history.pop(0)

    # ----------------------------------------------------------------
    def _get_average_color(self, track_id: int) -> Optional[np.ndarray]:
        """
        Bir track_id için biriken tüm renk örneklerinin ortalamasını döndürür.

        NOT: float64 olarak döndürüyoruz (renk örnekleri extract_jersey_color
        içinde float32 olarak toplanıyor). scikit-learn'ün KMeans.predict()
        metodu, fit() sırasında öğrendiği merkezlerle (genelde float64) aynı
        dtype'ı bekler -- aksi halde Cython arayüzünde "Buffer dtype mismatch"
        hatası fırlatır. Bu satırı kaldırırsan, model fit edildikten SONRA
        gelen her yeni oyuncuda predict() çağrısı çökebilir.
        """
        history = self.color_history.get(track_id)
        if not history:
            return None
        return np.mean(history, axis=0).astype(np.float64)

    # ----------------------------------------------------------------
    def _try_fit_kmeans(self):
        """
        Yeterli sayıda farklı oyuncudan renk örneği toplandıysa, global
        K-Means(n_clusters=3) modelini BİR KERE eğitir ve küme->rol
        eşlemesini (Team A / Team B / Referee) belirler.
        """
        if self.is_fitted:
            return  # zaten fit edilmiş, tekrar etme

        if len(self.color_history) < self.min_unique_tracks:
            return  # henüz yeterli çeşitlilikte oyuncu görülmedi

        # Her track_id için ortalama rengi bir özellik vektörü olarak kullan
        track_ids = list(self.color_history.keys())
        feature_matrix = np.array([self._get_average_color(tid) for tid in track_ids], dtype=np.float64)

        print(
            f"[TeamClassifier] {len(track_ids)} farklı oyuncudan yeterli renk örneği "
            f"toplandı. K-Means(n_clusters={self.n_clusters}) eğitiliyor..."
        )

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(feature_matrix)

        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        def _cluster_brightness(cluster_idx: int) -> float:
            center = self.kmeans.cluster_centers_[cluster_idx]
            if self.color_space.upper() == "LAB":
                return float(center[0])
            return float(center.sum())

        if self.n_clusters == 2:
            if len(cluster_sizes) < 2:
                print("[TeamClassifier] UYARI: Renk örnekleri en az 2 kümeye ayrışamadı; daha fazla veri bekleniyor.")
                return
            clusters_by_brightness = sorted(cluster_sizes.keys(), key=_cluster_brightness)
            self.cluster_role_map = {
                clusters_by_brightness[0]: "Team A",
                clusters_by_brightness[1]: "Team B",
            }
        else:
            if len(cluster_sizes) < 2:
                print("[TeamClassifier] UYARI: Renk örnekleri yeterli kümeye ayrışamadı; daha fazla veri bekleniyor.")
                return

            # n_clusters == 3: config.TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT=True
            # ile görüntüde ayrı bir hakem grubu OLDUĞUNU varsayıyoruz.
            # "En az üyeli küme = hakem" sezgisi kullanılır.
            referee_cluster = min(cluster_sizes, key=cluster_sizes.get)

            # --- GÜVENLİK KONTROLÜ ---
            other_sizes = [s for c, s in cluster_sizes.items() if c != referee_cluster]
            avg_other_size = sum(other_sizes) / len(other_sizes) if other_sizes else 0

            if avg_other_size > 0 and cluster_sizes[referee_cluster] >= avg_other_size * config.TEAM_CLASSIFIER_REFEREE_SIZE_RATIO_THRESHOLD:
                print(
                    f"\n[TeamClassifier] UYARI: En küçük küme ({cluster_sizes[referee_cluster]} oyuncu) "
                    f"diğer kümelere ({other_sizes}) göre yeterince küçük DEĞİL. Bu, görüntüde GERÇEK bir "
                    f"hakem grubu OLMAYABİLECEĞİNİ gösteriyor -- K-Means muhtemelen sadece kusurlu bir "
                    f"renk ayrımı yaptı ve bazı GERÇEK OYUNCULARI yanlışlıkla 'Hakem' etiketleyecek.\n"
                    f"                 Eğer görüntüde net, ayırt edici renkli bir hakem YOKSA, "
                    f"config.py içinde şunu ayarla: TEAM_CLASSIFIER_ASSUME_REFEREE_PRESENT = False\n"
                )

            remaining_clusters = [c for c in cluster_sizes if c != referee_cluster]
            # Kümeleri deterministik olsun diye parlaklığa göre sıralıyoruz.
            remaining_clusters.sort(key=_cluster_brightness)

            if len(remaining_clusters) < 2:
                # Örnekler yalnızca 2 belirgin renk kümesine çöktüyse güvenli biçimde Team A / Team B ata
                clusters_by_brightness = sorted(cluster_sizes.keys(), key=_cluster_brightness)
                self.cluster_role_map = {
                    clusters_by_brightness[0]: "Team A",
                    clusters_by_brightness[1]: "Team B",
                }
            else:
                self.cluster_role_map = {
                    remaining_clusters[0]: "Team A",
                    remaining_clusters[1]: "Team B",
                    referee_cluster: "Referee",
                }

        print(f"[TeamClassifier] Küme boyutları: {cluster_sizes} -> Rol eşlemesi: {self.cluster_role_map}")

        # Şimdiye kadar toplanan tüm oyunculara nihai etiketlerini ata, önbellekle
        # ve renk geçmişini bellekten temizle.
        for tid, label in zip(track_ids, cluster_labels):
            self.team_assignment[tid] = self.cluster_role_map[label]
            self.color_history.pop(tid, None)

        self.is_fitted = True

    # ----------------------------------------------------------------
    def assign_teams(self, tracked_players: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Bir karedeki takip edilmiş oyuncu listesini alır, her birine "team"
        alanını doldurup geri döndürür.

        NOT: Top zaten forma giymediği için bu sınıfın kapsamı dışındadır --
        çağıran main.py zaten sadece tracked_players'ı (top hariç) burada geçirir.

        Fine-tune futbol modeli kullanılıyorsa (config.USES_FINE_TUNED_FOOTBALL_MODEL),
        class_name == "referee" olan tespitler renk kümelemesine HİÇ sokulmadan
        doğrudan "Referee" etiketlenir (bkz. modül başındaki açıklama).

        Parametreler
        ------------
        tracked_players : List[Dict]
            tracker.py'den gelen, track_id doldurulmuş oyuncu tespitleri.
        frame : np.ndarray
            Mevcut kare (forma rengini piksellerden okuyabilmek için gerekli).

        Dönüş
        -----
        List[Dict]
            Aynı liste, ama her elemanın "team" alanı artık "Team A" / "Team B" /
            "Referee" ya da (henüz yeterli veri toplanmadıysa) None olabilir.
        """
        for det in tracked_players:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            # --- 0) Fine-tune modelde hakem ve kaleci zaten YOLO sınıfından biliniyor --
            # renk kümelemesine hiç sokmadan doğrudan etiketle ve bir sonraki
            # tespite geç (bu track_id'ler hiçbir zaman color_history'ye girmez). ---
            if config.USES_FINE_TUNED_FOOTBALL_MODEL:
                cls_name = det.get("class_name")
                if cls_name == "referee":
                    det["team"] = "Referee"
                    self.team_assignment[track_id] = "Referee"
                    self.color_history.pop(track_id, None)
                    continue
                if cls_name == "goalkeeper":
                    det["team"] = "Goalkeeper"
                    self.team_assignment[track_id] = "Goalkeeper"
                    self.color_history.pop(track_id, None)
                    continue

            # --- 1) ÖNBELLEK KONTROLÜ (KRİTİK HIZ OPTİMİZASYONU) ---
            # Bu oyuncu için daha önce nihai karar verildiyse bir daha
            # extract_jersey_color / K-Means çalıştırmadan doğrudan etiketi kullan.
            if track_id in self.team_assignment:
                det["team"] = self.team_assignment[track_id]
                continue

            # --- 2) Henüz sabitlenmediyse bu karedeki forma rengini çıkar ve geçmişe ekle ---
            color = self.extract_jersey_color(frame, det["bbox"])
            if color is not None:
                self._collect_sample(track_id, color)

            # --- 3) Henüz kesin karar yoksa, model fit edilmeye hazır mı diye dene ---
            self._try_fit_kmeans()

            if track_id in self.team_assignment:
                # _try_fit_kmeans() az önce bu oyuncuyu da etiketlemiş olabilir
                det["team"] = self.team_assignment[track_id]
            elif self.is_fitted:
                # Model zaten fit edilmiş ama bu track_id henüz kilitlenmedi --
                # mevcut ortalama rengiyle SADECE predict yap; min_samples_for_lock
                # eşiğine ulaşınca önbelleğe kilitle.
                avg_color = self._get_average_color(track_id)
                if avg_color is not None:
                    predicted_cluster = self.kmeans.predict([avg_color])[0]
                    label = self.cluster_role_map[predicted_cluster]
                    det["team"] = label
                    history = self.color_history.get(track_id)
                    if history is not None and len(history) >= self.min_samples_for_lock:
                        self.team_assignment[track_id] = label
                        self.color_history.pop(track_id, None)
                else:
                    det["team"] = None
            else:
                # Model henüz fit edilmedi (yeterli çeşitlilik yok) -- takım bilgisi yok
                det["team"] = None

        # Aynı kare içinde K-Means ilk kez fit edildiyse, döngünün başındaki
        # oyuncuların etiketlerini de yeni dolan team_assignment'tan tamamla.
        if self.is_fitted:
            for det in tracked_players:
                if det.get("team") is None:
                    tid = det.get("track_id")
                    if tid in self.team_assignment:
                        det["team"] = self.team_assignment[tid]

        return tracked_players

    # ----------------------------------------------------------------
    def get_team_map(self) -> Dict[int, str]:
        """
        Bilinen tüm track_id -> team eşlemelerini döndürür. Henüz min_samples_for_lock
        eşiğini doldurmamış ama model fit edilmiş track'leri de predict ederek ekler.
        """
        team_map = dict(self.team_assignment)
        if self.is_fitted and self.kmeans is not None:
            for track_id in self.color_history:
                if track_id not in team_map:
                    avg_color = self._get_average_color(track_id)
                    if avg_color is not None:
                        predicted_cluster = self.kmeans.predict([avg_color])[0]
                        team_map[track_id] = self.cluster_role_map[predicted_cluster]
        return team_map


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa Adım 1+2+3+4'ü uçtan uca test eder.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modules.video_processor import VideoProcessor
    from modules.detector import ObjectDetector
    from modules.tracker import PlayerTracker, BallTracker

    detector = ObjectDetector()
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    team_classifier = TeamClassifier()

    with VideoProcessor(config.DEFAULT_INPUT_VIDEO, config.DEFAULT_OUTPUT_VIDEO) as vp:
        for frame_idx, frame in vp.read_frames():
            raw_detections = detector.detect_frame(frame)
            tracked_players = player_tracker.update(raw_detections, frame)
            tracked_ball = ball_tracker.update(raw_detections)

            # Adım 4: forma rengine göre takım ata
            tracked_players = team_classifier.assign_teams(tracked_players, frame)

            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            annotated_frame = vp.draw_detections(frame, all_tracked)
            vp.write_frame(annotated_frame)

    print("Takım sınıflandırma testi tamamlandı. Oyuncular artık takım rengine göre boyanmış olmalı.")
