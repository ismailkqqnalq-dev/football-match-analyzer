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
    ):
        self.n_clusters = n_clusters
        self.min_unique_tracks = min_unique_tracks
        self.max_history_per_track = max_history_per_track

        # track_id -> [renk_örneği_1, renk_örneği_2, ...] (her biri 3 elemanlı BGR array)
        self.color_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # track_id -> "Team A" / "Team B" / "Referee"  (bir kere hesaplanınca sabitlenir)
        self.team_assignment: Dict[int, str] = {}

        # K-Means modelinin kendisi -- yeterli veri toplanınca bir kere fit edilir
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False

        # kmeans küme indeksi (0,1,2) -> "Team A"/"Team B"/"Referee" eşlemesi
        self.cluster_role_map: Dict[int, str] = {}

        print(
            f"[TeamClassifier] Oluşturuldu. En az {min_unique_tracks} farklı oyuncu "
            f"görülünce K-Means kümeleme otomatik olarak devreye girecek."
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

        # Pikselleri (N, 3) şekline getir -- her satır bir pikselin BGR değeri
        pixels = jersey_crop.reshape(-1, 3).astype(np.float32)

        # --- 3) 2 kümeli K-Means: forma pikselleri vs arka plan pikselleri ---
        # n_init=3 -> farklı başlangıç noktalarıyla 3 kez deneyip en iyisini seç
        # (küçük veri setinde ucuz bir işlem, kararlılığı artırır)
        pixel_kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
        pixel_labels = pixel_kmeans.fit_predict(pixels)

        # --- 4) Arka plan kümesini köşe piksellerinden tahmin et ---
        # jersey_crop 30x30 olduğu için köşe indeksleri: (0,0), (0,29), (29,0), (29,29)
        labels_2d = pixel_labels.reshape(30, 30)
        corner_labels = [
            labels_2d[0, 0], labels_2d[0, -1],
            labels_2d[-1, 0], labels_2d[-1, -1],
        ]
        # Köşelerde en sık görülen küme etiketi = muhtemelen arka plan (çim/tribün)
        background_cluster = max(set(corner_labels), key=corner_labels.count)
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
        history = self.color_history[track_id]
        history.append(color)
        # Limit aşılırsa en eski örneği at (sliding window mantığı)
        if len(history) > self.max_history_per_track:
            history.pop(0)

    # ----------------------------------------------------------------
    def _get_average_color(self, track_id: int) -> Optional[np.ndarray]:
        """Bir track_id için biriken tüm renk örneklerinin ortalamasını döndürür."""
        history = self.color_history.get(track_id)
        if not history:
            return None
        return np.mean(history, axis=0)

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
        feature_matrix = np.array([self._get_average_color(tid) for tid in track_ids])

        print(
            f"[TeamClassifier] {len(track_ids)} farklı oyuncudan yeterli renk örneği "
            f"toplandı. K-Means(n_clusters={self.n_clusters}) eğitiliyor..."
        )

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(feature_matrix)

        # --- Hangi küme "Hakem"? En az üyeli küme (sahada en az kişi olan grup) ---
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
        referee_cluster = min(cluster_sizes, key=cluster_sizes.get)

        remaining_clusters = [c for c in cluster_sizes if c != referee_cluster]
        # Team A / Team B ayrımı keyfi olabilir (hangisinin "A" hangisinin "B"
        # olduğunun taktiksel bir önemi yok, sadece tutarlı olması önemli).
        # Kümeleri deterministik olsun diye küme merkezinin toplam parlaklığına
        # göre sıralıyoruz (koyu forma -> Team A, açık forma -> Team B gibi sabit bir kural).
        remaining_clusters.sort(key=lambda c: self.kmeans.cluster_centers_[c].sum())

        self.cluster_role_map = {
            remaining_clusters[0]: "Team A",
            remaining_clusters[1]: "Team B",
            referee_cluster: "Referee",
        }

        print(f"[TeamClassifier] Küme boyutları: {cluster_sizes} -> Rol eşlemesi: {self.cluster_role_map}")

        # Şimdiye kadar toplanan tüm oyunculara nihai etiketlerini ata ve önbellekle
        for tid, label in zip(track_ids, cluster_labels):
            self.team_assignment[tid] = self.cluster_role_map[label]

        self.is_fitted = True

    # ----------------------------------------------------------------
    def assign_teams(self, tracked_players: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Bir karedeki takip edilmiş oyuncu listesini alır, her birine "team"
        alanını doldurup geri döndürür.

        NOT: Bu metod SADECE class_name == "person" olan (yani track_id'si
        DeepSORT'tan gelen) tespitler için anlamlıdır. Top zaten forma
        giymediği için bu sınıfın kapsamı dışındadır.

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

            # --- 1) Bu karedeki forma rengini çıkar ve geçmişe ekle ---
            color = self.extract_jersey_color(frame, det["bbox"])
            if color is not None:
                self._collect_sample(track_id, color)

            # --- 2) Eğer bu oyuncu için daha önce nihai bir takım kararı verildiyse,
            #        onu doğrudan kullan (kararlılık için -- tekrar hesaplamıyoruz) ---
            if track_id in self.team_assignment:
                det["team"] = self.team_assignment[track_id]
                continue

            # --- 3) Henüz kesin karar yoksa, model fit edilmeye hazır mı diye dene ---
            self._try_fit_kmeans()

            if track_id in self.team_assignment:
                # _try_fit_kmeans() az önce bu oyuncuyu da etiketlemiş olabilir
                det["team"] = self.team_assignment[track_id]
            elif self.is_fitted:
                # Model zaten fit edilmiş ama bu track_id yeni görülüyor (henüz
                # geçmişte yoktu) -- mevcut ortalama rengiyle SADECE predict yap,
                # yeniden fit ETME. Sonucu bu track_id için önbellekle.
                avg_color = self._get_average_color(track_id)
                if avg_color is not None:
                    predicted_cluster = self.kmeans.predict([avg_color])[0]
                    label = self.cluster_role_map[predicted_cluster]
                    self.team_assignment[track_id] = label
                    det["team"] = label
            else:
                # Model henüz fit edilmedi (yeterli çeşitlilik yok) -- takım bilgisi yok
                det["team"] = None

        return tracked_players


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
