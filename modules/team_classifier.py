"""
team_classifier.py
-------------------
ADIM 4: TAKIM SINIFLANDIRMASI (K-MEANS CLUSTERING + ÇOK RENKLİ / GÖLGE DAYANIKLI TEMSİL)

Bu modülün görevi: her oyuncunun forma görünümüne bakarak onu Takım A, Takım B
veya Hakem olarak etiketlemek; kaleciyi de kendi yarı sahasındaki savunma bloğunun
konumuna göre ilgili takımla (Team A / Team B) eşleştirmektir.

GENEL STRATEJİ (Faz A1):
    1) Her tespit edilen oyuncu kutusundan gövde bölgesi kırpılır ve 3-kümeli
       (köşe arka plan ayrıştırmalı) + çim kenar temizlemeli yöntemle forma
       pikselleri ayrılır; böylece çubuklu/parçalı (2 renkli) formalar arka
       planla veya tek renge çökerek kaybolmaz.
    2) Çıkarılan özellik vektörü yalnızca tek bir 3B LAB ortalaması değil;
       gölgeye dayanıklı normalize RGB (r, g) 2B kromatik histogram, LAB (a, b)
       2B kromatik histogram, normalize parlaklık (L) dağılımı ve özet
       istatistiklerden oluşan sabit uzunluklu (60 boyutlu float32) bir
       betimleyicidir.
    3) Yeterli sayıda farklı oyuncudan (config.TEAM_CLASSIFIER_MIN_UNIQUE_TRACKS)
       örnek toplanınca K-Means eğitilir. İlk kilitleme sonrası kilitli izler
       her `refresh_interval` gözlemde yenilenir, hareketli çoğunluk oylaması
       (`_vote_history`) ile hatalı ilk kilitlemeler düzeltilir ve erken dönemde
       sınırlı sayıda küme merkezi iyileştirmesi (`_maybe_refit_kmeans`) ile
       küme sınırları rafine edilir (küme etiketleri minimum merkez mesafesiyle
       hizalanarak Team A / Team B kimlikleri korunur).
    4) Kaleci (`class_name == "goalkeeper"`): Model eğitilmiş ve iki takımın
       saha içi yatay konumları belirgin biçimde ayrışmışsa, kaleci kendi
       savunma bloğuna en yakın takımla ("Team A" veya "Team B") eşleştirilir
       (devre arası taraf değişimlerine uyum için EMA penceresi kullanılır).
       Henüz model eğitilmemiş veya takımlar yatayda ayrışmamışsa güvenli
       geri dönüş olarak geçici `"Goalkeeper"` döner; `det["class_name"]`
       hiçbir zaman değiştirilmez.
"""

import itertools
import os
import sys
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class TeamClassifier:
    """
    Forma rengine/desenine ve gölgeye dayanıklı temsile göre oyuncuları
    Takım A / Takım B / Hakem olarak sınıflandıran ve kalecileri savunma
    bloğuna göre takımlara eşleştiren sınıf.
    """

    FEATURE_DIM = 60  # 4 özet + 36 (r,g) hist + 16 (a,b) hist + 4 L hist

    def __init__(
        self,
        n_clusters: int = config.TEAM_CLASSIFIER_N_CLUSTERS,
        min_unique_tracks: int = config.TEAM_CLASSIFIER_MIN_UNIQUE_TRACKS,
        max_history_per_track: int = config.TEAM_COLOR_HISTORY_MAX_SAMPLES,
        color_space: str = config.TEAM_COLOR_SPACE,
        min_samples_for_lock: int = config.TEAM_CLASSIFIER_MIN_SAMPLES_FOR_LOCK,
        refresh_interval: int = getattr(config, "TEAM_CLASSIFIER_LOCK_REFRESH_INTERVAL", 5),
        vote_window: int = getattr(config, "TEAM_CLASSIFIER_VOTE_WINDOW", 7),
        max_refits: int = getattr(config, "TEAM_CLASSIFIER_MAX_REFITS", 3),
    ):
        self.n_clusters = n_clusters
        self.min_unique_tracks = min_unique_tracks
        self.max_history_per_track = max_history_per_track
        self.color_space = color_space
        self.min_samples_for_lock = max(1, min_samples_for_lock)
        self.refresh_interval = max(1, int(refresh_interval))
        self.vote_window = max(3, int(vote_window))
        self.max_refits = max(0, int(max_refits))

        # Kilitlenmemiş izler için geçici ön-örnek tamponu (kilitlenince pop edilir!)
        self.color_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # track_id -> "Team A" / "Team B" / "Referee" (kararlı önbellek)
        self.team_assignment: Dict[int, str] = {}

        # Post-lock hareketli oylama ve sınırlı model rafinasyonu tamponları
        self._track_seen_count: Dict[int, int] = {}
        self._vote_history: Dict[int, Deque[str]] = {}
        self._track_feature_bank: Dict[int, np.ndarray] = {}
        self._refit_count: int = 0
        self._samples_since_fit: int = 0

        # Kaleci mekânsal eşleştirme ve kilitli çoğunluk oylaması durumu
        self._team_x_ema: Dict[str, float] = {}
        self._gk_vote_history: Dict[int, Deque[str]] = {}
        self._locked_gk_tracks: set = set()

        # K-Means modeli ve küme->rol haritası
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False
        self.cluster_role_map: Dict[int, str] = {}

        print(
            f"[TeamClassifier] Oluşturuldu (color_space={self.color_space}). "
            f"En az {min_unique_tracks} farklı oyuncu görülünce K-Means kümeleme "
            f"otomatik olarak devreye girecek."
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _separate_jersey_foreground(
        pixels_cs: np.ndarray,
        pixels_bgr: np.ndarray,
        crop_h: int = 30,
        crop_w: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministik 3-kümeli köşe-arka-plan ayrıştırması:
        Çubuklu/parçalı formalardaki iki farklı forma renginin tek arka plan
        kümesiyle karışmasını önler.
        """
        n_pix = len(pixels_cs)
        if n_pix < 9:
            return pixels_cs, pixels_bgr

        uniq_cnt = len(np.unique(pixels_cs.astype(np.int32), axis=0))
        if uniq_cnt < 2:
            return pixels_cs, pixels_bgr

        k_px = 3 if uniq_cnt >= 3 else 2

        idx_grid = np.arange(n_pix, dtype=np.int32).reshape(crop_h, crop_w)
        c_h = min(3, max(1, crop_h // 4))
        c_w = min(3, max(1, crop_w // 4))
        corner_idx = np.concatenate([
            idx_grid[:c_h, :c_w].ravel(),
            idx_grid[:c_h, -c_w:].ravel(),
            idx_grid[-c_h:, :c_w].ravel(),
            idx_grid[-c_h:, -c_w:].ravel(),
        ])

        # Deterministik en-uzak-nokta başlatma (köşe medyanından başlayarak)
        centers = np.empty((k_px, 3), dtype=np.float32)
        centers[0] = np.median(pixels_cs[corner_idx], axis=0)
        d_min = np.sum((pixels_cs - centers[0]) ** 2, axis=1)
        centers[1] = pixels_cs[int(np.argmax(d_min))]
        if k_px == 3:
            d1 = np.sum((pixels_cs - centers[1]) ** 2, axis=1)
            d_min = np.minimum(d_min, d1)
            centers[2] = pixels_cs[int(np.argmax(d_min))]

        labels = np.zeros(n_pix, dtype=np.int32)
        for _ in range(6):
            dists = np.sum((pixels_cs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_i in range(k_px):
                cm = labels == c_i
                if cm.any():
                    centers[c_i] = pixels_cs[cm].mean(axis=0)

        corner_labels = labels[corner_idx]
        corner_counts = np.bincount(corner_labels, minlength=k_px)
        bg_primary = int(np.argmax(corner_counts))
        bg_mask = labels == bg_primary

        if k_px == 3:
            for c_i in range(k_px):
                if c_i != bg_primary and corner_counts[c_i] >= 0.30 * len(corner_idx):
                    cand_bg = bg_mask | (labels == c_i)
                    if int((~cand_bg).sum()) >= max(30, int(0.15 * n_pix)):
                        bg_mask = cand_bg

        fg_mask = ~bg_mask
        if int(fg_mask.sum()) < max(25, int(0.08 * n_pix)):
            return pixels_cs, pixels_bgr

        fg_cs = pixels_cs[fg_mask]
        fg_bgr = pixels_bgr[fg_mask]

        # Köşeler çim yeşiliyse, kenarlardan sızan artık yeşil pikselleri temizle
        corner_bgr = pixels_bgr[corner_idx]
        c_med_bgr = np.median(corner_bgr, axis=0)
        corners_are_green = (c_med_bgr[1] > c_med_bgr[2] + 15.0) and (c_med_bgr[1] > c_med_bgr[0] + 15.0)
        if corners_are_green:
            green_fg = (fg_bgr[:, 1] > fg_bgr[:, 2] + 20.0) & (fg_bgr[:, 1] > fg_bgr[:, 0] + 20.0)
            non_green = ~green_fg
            if int(non_green.sum()) >= max(20, int(0.25 * len(fg_bgr))):
                fg_cs = fg_cs[non_green]
                fg_bgr = fg_bgr[non_green]

        return fg_cs, fg_bgr

    # ----------------------------------------------------------------
    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Bir oyuncunun bounding box'ından çubuklu/parçalı formalara ve stadyum
        gölgesine dayanıklı 60 boyutlu (float32) forma görünüm betimleyicisi çıkarır.

        İçerik:
          - [0..3]: Gölge ölçekli ortalama parlaklık L (indeks 0 -> Team A/B sırası),
                    ortalama normalize r, g ve parlaklık yayılımı
          - [4..39]: 6x6 2B normalize RGB (r, g) kromatik histogram (parçalı/çubuklu + gölgeye dayanıklı)
          - [40..55]: 4x4 2B LAB (a, b) kromatik histogram
          - [56..59]: 4 kutulu normalize parlaklık (L) dağılımı (akromatik beyaz/siyah ayrımı için)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        box_height = y2 - y1
        box_width = x2 - x1

        if box_height < 10 or box_width < 10:
            return None

        crop_y1 = y1 + int(box_height * config.JERSEY_CROP_TOP_RATIO)
        crop_y2 = y1 + int(box_height * config.JERSEY_CROP_BOTTOM_RATIO)
        crop_x1 = x1 + int(box_width * config.JERSEY_CROP_LEFT_RATIO)
        crop_x2 = x1 + int(box_width * config.JERSEY_CROP_RIGHT_RATIO)

        jersey_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if jersey_crop_bgr.size == 0:
            return None

        jersey_crop_bgr = cv2.resize(jersey_crop_bgr, (30, 30), interpolation=cv2.INTER_AREA)
        jersey_crop_lab = cv2.cvtColor(jersey_crop_bgr, cv2.COLOR_BGR2LAB)

        pixels_bgr = jersey_crop_bgr.reshape(-1, 3).astype(np.float32)
        pixels_lab = jersey_crop_lab.reshape(-1, 3).astype(np.float32)
        pixels_cs = pixels_lab if self.color_space.upper() == "LAB" else pixels_bgr

        _, fg_bgr = self._separate_jersey_foreground(pixels_cs, pixels_bgr, 30, 30)
        # Ayrılmış ön plan pikselleri için karşılık gelen LAB değerlerini hesapla
        if len(fg_bgr) == len(pixels_bgr):
            fg_lab = pixels_lab
        else:
            fg_lab = cv2.cvtColor(
                np.clip(fg_bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                cv2.COLOR_BGR2LAB,
            ).reshape(-1, 3).astype(np.float32)

        n_fg = float(len(fg_bgr))
        if n_fg <= 0:
            return None

        # 1) Parlaklık L [0..1]
        if self.color_space.upper() == "LAB":
            l_norm = np.clip(fg_lab[:, 0] / 255.0, 0.0, 0.9999)
        else:
            l_norm = np.clip(fg_bgr.mean(axis=1) / 255.0, 0.0, 0.9999)

        # 2) Gölge-değişmez normalize RGB (r, g) kromatiklik
        rgb_sum = fg_bgr[:, 2] + fg_bgr[:, 1] + fg_bgr[:, 0] + 1e-3
        r_ch = np.clip(fg_bgr[:, 2] / rgb_sum, 0.0, 0.9999)
        g_ch = np.clip(fg_bgr[:, 1] / rgb_sum, 0.0, 0.9999)

        r_bin = (r_ch * 6.0).astype(np.int32)
        g_bin = (g_ch * 6.0).astype(np.int32)
        rg_hist = (np.bincount(r_bin * 6 + g_bin, minlength=36).astype(np.float32) / n_fg) * 1.20

        # 3) LAB (a, b) 4x4 histogram [64..192]
        a_norm = np.clip((fg_lab[:, 1] - 64.0) / 128.0, 0.0, 0.9999)
        b_norm = np.clip((fg_lab[:, 2] - 64.0) / 128.0, 0.0, 0.9999)
        a_bin = (a_norm * 4.0).astype(np.int32)
        b_bin = (b_norm * 4.0).astype(np.int32)
        ab_hist = (np.bincount(a_bin * 4 + b_bin, minlength=16).astype(np.float32) / n_fg) * 0.15

        # 4) 4 kutulu normalize parlaklık dağılımı (akromatik beyaz/siyah ayrımı için)
        l_bin = (l_norm * 4.0).astype(np.int32)
        l_hist = (np.bincount(l_bin, minlength=4).astype(np.float32) / n_fg) * 0.25

        # 5) Özet özellikler (indeks 0: ölçekli ortalama parlaklık)
        summary = np.array([
            float(l_norm.mean()) * 0.25,
            float(r_ch.mean()) * 0.60,
            float(g_ch.mean()) * 0.60,
            float(l_norm.std()) * 0.15,
        ], dtype=np.float32)

        return np.concatenate([summary, rg_hist, ab_hist, l_hist]).astype(np.float32)

    # ----------------------------------------------------------------
    def _collect_sample(self, track_id: int, color: np.ndarray):
        """Bir oyuncunun renk/görünüm betimleyicisini ön-örnek geçmişine ekler."""
        if self.is_fitted and track_id not in self.color_history:
            max_unlocked = int(getattr(config, "TEAM_COLOR_HISTORY_MAX_UNLOCKED_TRACKS", 512))
            if max_unlocked > 0:
                while len(self.color_history) >= max_unlocked:
                    oldest_tid = next(iter(self.color_history))
                    self.color_history.pop(oldest_tid, None)
        history = self.color_history[track_id]
        history.append(color)
        if len(history) > self.max_history_per_track:
            history.pop(0)

    # ----------------------------------------------------------------
    def _store_track_feature_bank(self, track_id: int, feat: np.ndarray):
        """Model rafinasyonu için track başına sınırlı EMA özellik bankasını günceller."""
        max_bank = int(getattr(config, "TEAM_CLASSIFIER_MAX_FEATURE_BANK", 512))
        if track_id in self._track_feature_bank:
            prev = self._track_feature_bank[track_id]
            if prev.shape == feat.shape:
                self._track_feature_bank[track_id] = (0.7 * prev + 0.3 * feat).astype(np.float64)
            else:
                self._track_feature_bank[track_id] = feat.astype(np.float64)
        else:
            if max_bank > 0 and len(self._track_feature_bank) >= max_bank:
                oldest_tid = next(iter(self._track_feature_bank))
                self._track_feature_bank.pop(oldest_tid, None)
            self._track_feature_bank[track_id] = feat.astype(np.float64)

    # ----------------------------------------------------------------
    def _record_vote_and_maybe_update(self, track_id: int, pred_label: str):
        """Kilitli iz için hareketli çoğunluk oylaması penceresini günceller."""
        max_votes = int(getattr(config, "TEAM_CLASSIFIER_MAX_TRACKED_VOTES", 2048))
        if track_id not in self._vote_history:
            if max_votes > 0 and len(self._vote_history) >= max_votes:
                oldest_tid = next(iter(self._vote_history))
                self._vote_history.pop(oldest_tid, None)
            self._vote_history[track_id] = deque(maxlen=self.vote_window)

        votes = self._vote_history[track_id]
        votes.append(pred_label)

        if len(votes) >= self.min_samples_for_lock:
            counts = Counter(votes)
            top_label, top_cnt = counts.most_common(1)[0]
            min_agree = float(getattr(config, "TEAM_CLASSIFIER_VOTE_MIN_AGREEMENT", 0.6))
            if (
                top_label != self.team_assignment.get(track_id)
                and (top_cnt / len(votes)) >= min_agree
                and top_cnt >= self.min_samples_for_lock
            ):
                self.team_assignment[track_id] = top_label

    # ----------------------------------------------------------------
    def _get_average_color(self, track_id: int) -> Optional[np.ndarray]:
        """Bir track_id için biriken örneklerin ortalamasını float64 döndürür."""
        history = self.color_history.get(track_id)
        if not history:
            return None
        return np.mean(history, axis=0).astype(np.float64)

    # ----------------------------------------------------------------
    def _try_fit_kmeans(self):
        """
        Yeterli sayıda farklı oyuncudan örnek toplandıysa global K-Means modelini
        eğitir ve küme->rol eşlemesini (Team A / Team B / Referee) belirler.
        """
        if self.is_fitted:
            return

        if len(self.color_history) < self.min_unique_tracks:
            return

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
            return float(center[0])

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

            referee_cluster = min(cluster_sizes, key=cluster_sizes.get)
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
            remaining_clusters.sort(key=_cluster_brightness)

            if len(remaining_clusters) < 2:
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

        for tid, label, feat in zip(track_ids, cluster_labels, feature_matrix):
            role = self.cluster_role_map[int(label)]
            self.team_assignment[tid] = role
            self._store_track_feature_bank(tid, feat)
            self._vote_history[tid] = deque([role], maxlen=self.vote_window)
            self._track_seen_count[tid] = 0
            self.color_history.pop(tid, None)

        self.is_fitted = True
        self._samples_since_fit = 0

    # ----------------------------------------------------------------
    def _maybe_refit_kmeans(self):
        """
        Maç başında çarpık/gölgeli ilk örneklerle eğitilmiş K-Means merkezlerini,
        yeni ve taze oyuncu örnekleri geldikçe sınırlı sayıda rafine eder.
        Küme rolleri minimum merkez mesafesi permütasyonu ile hizalanır, böylece
        'Team A' / 'Team B' kimlikleri sahne kesmeleri veya maç boyunca ters dönmez.
        """
        if not self.is_fitted or self.kmeans is None or self._refit_count >= self.max_refits:
            return
        min_bank = max(self.min_unique_tracks + 4, int(self.min_unique_tracks * 1.4))
        if len(self._track_feature_bank) < min_bank:
            return
        if self._samples_since_fit < max(6, self.min_unique_tracks // 2):
            return

        tids = list(self._track_feature_bank.keys())
        mat = np.array([self._track_feature_bank[tid] for tid in tids], dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] < self.n_clusters or mat.shape[1] != self.kmeans.cluster_centers_.shape[1]:
            return

        new_km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        new_labels = new_km.fit_predict(mat)
        if len(np.unique(new_labels)) < self.n_clusters:
            return

        old_centers = self.kmeans.cluster_centers_
        new_centers = new_km.cluster_centers_

        # Eski ve yeni küme merkezlerini minimum toplam Öklid mesafesiyle eşle
        best_perm = tuple(range(self.n_clusters))
        best_dist = float("inf")
        for perm in itertools.permutations(range(self.n_clusters)):
            d = sum(
                float(np.linalg.norm(new_centers[c] - old_centers[perm[c]]))
                for c in range(self.n_clusters)
            )
            if d < best_dist:
                best_dist = d
                best_perm = perm

        new_role_map = {
            c: self.cluster_role_map.get(best_perm[c], f"Cluster_{c}")
            for c in range(self.n_clusters)
        }
        self.kmeans = new_km
        self.cluster_role_map = new_role_map

        # Mevcut kilitli izleri güncellenmiş karar sınırına göre yeniden değerlendir
        for tid, cl_idx in zip(tids, new_labels):
            if tid in self._locked_gk_tracks:
                continue
            if tid in self.team_assignment:
                cur_role = self.team_assignment[tid]
                if cur_role not in ("Team A", "Team B", "Referee"):
                    continue
                new_role = self.cluster_role_map[int(cl_idx)]
                if new_role != cur_role:
                    self.team_assignment[tid] = new_role
                    self._vote_history[tid] = deque([new_role], maxlen=self.vote_window)

        self._refit_count += 1
        self._samples_since_fit = 0

    # ----------------------------------------------------------------
    def _update_team_spatial_ema(self, outfield_dets: List[Dict]):
        """Her iki takımın saha oyuncuları birlikte görüldüğünde yatay x EMA'sını günceller."""
        xs_by_team: Dict[str, List[float]] = {}
        for team_label in ("Team A", "Team B"):
            xs = [
                0.5 * (float(d["bbox"][0]) + float(d["bbox"][2]))
                for d in outfield_dets
                if d.get("team") == team_label and d.get("bbox") is not None and len(d["bbox"]) >= 4
            ]
            if not xs:
                return
            xs_by_team[team_label] = xs

        alpha = float(getattr(config, "TEAM_CLASSIFIER_GK_X_EMA_ALPHA", 0.15))
        for team_label, xs in xs_by_team.items():
            cur_mean = float(np.mean(xs))
            if team_label not in self._team_x_ema:
                self._team_x_ema[team_label] = cur_mean
            else:
                self._team_x_ema[team_label] = (1.0 - alpha) * self._team_x_ema[team_label] + alpha * cur_mean

    # ----------------------------------------------------------------
    def _resolve_goalkeeper_team(
        self,
        gk_bbox: List[float],
        outfield_dets: List[Dict],
    ) -> str:
        """
        Model eğitilmiş ve saha oyuncuları yatayda yeterince ayrışmışsa kaleciyi
        en yakın savunma bloğunun takımıyla eşleştirir; aksi halde geçici 'Goalkeeper' döner.
        """
        if not self.is_fitted:
            return "Goalkeeper"
        if "Team A" not in self._team_x_ema or "Team B" not in self._team_x_ema:
            return "Goalkeeper"

        ema_a = self._team_x_ema["Team A"]
        ema_b = self._team_x_ema["Team B"]

        xs_a = sorted(
            0.5 * (float(d["bbox"][0]) + float(d["bbox"][2]))
            for d in outfield_dets
            if d.get("team") == "Team A" and d.get("bbox") is not None and len(d["bbox"]) >= 4
        )
        xs_b = sorted(
            0.5 * (float(d["bbox"][0]) + float(d["bbox"][2]))
            for d in outfield_dets
            if d.get("team") == "Team B" and d.get("bbox") is not None and len(d["bbox"]) >= 4
        )

        # Savunma bloğu merkezi: soldaki takımın en derin %50'si, sağdaki takımın en derin %50'si
        if ema_a <= ema_b:
            cur_a = float(np.mean(xs_a[:max(1, (len(xs_a) + 1) // 2)])) if xs_a else ema_a
            cur_b = float(np.mean(xs_b[len(xs_b) // 2:])) if xs_b else ema_b
        else:
            cur_a = float(np.mean(xs_a[len(xs_a) // 2:])) if xs_a else ema_a
            cur_b = float(np.mean(xs_b[:max(1, (len(xs_b) + 1) // 2)])) if xs_b else ema_b

        block_a = 0.5 * cur_a + 0.5 * ema_a if xs_a else ema_a
        block_b = 0.5 * cur_b + 0.5 * ema_b if xs_b else ema_b

        min_sep_px = float(getattr(config, "TEAM_CLASSIFIER_GK_MIN_SEPARATION_PX", 40.0))
        if abs(block_a - block_b) < min_sep_px and abs(ema_a - ema_b) < min_sep_px:
            return "Goalkeeper"

        gk_x = 0.5 * (float(gk_bbox[0]) + float(gk_bbox[2]))
        return "Team A" if abs(gk_x - block_a) <= abs(gk_x - block_b) else "Team B"

    # ----------------------------------------------------------------
    def assign_teams(self, tracked_players: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Bir karedeki takip edilmiş oyuncu listesini alır, her birine "team"
        alanını doldurup geri döndürür (`det["class_name"]` asla değiştirilmez).
        """
        gk_dets: List[Dict] = []
        outfield_dets: List[Dict] = []

        for det in tracked_players:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            cls_name = det.get("class_name")
            if config.USES_FINE_TUNED_FOOTBALL_MODEL:
                if cls_name == "referee":
                    det["team"] = "Referee"
                    self.team_assignment[track_id] = "Referee"
                    self.color_history.pop(track_id, None)
                    continue
                if cls_name == "goalkeeper":
                    self.color_history.pop(track_id, None)
                    gk_dets.append(det)
                    continue
            elif cls_name == "goalkeeper":
                self.color_history.pop(track_id, None)
                gk_dets.append(det)
                continue

            outfield_dets.append(det)

            # --- 1) ÖNBELLEK KONTROLÜ + PERİYODİK YENİLEME / ÇOĞUNLUK OYLAMASI ---
            if track_id in self.team_assignment:
                seen = self._track_seen_count.get(track_id, 0) + 1
                self._track_seen_count[track_id] = seen

                if (
                    self.is_fitted
                    and self.kmeans is not None
                    and (seen % self.refresh_interval == 0)
                ):
                    color = self.extract_jersey_color(frame, det["bbox"])
                    if color is not None:
                        self._store_track_feature_bank(track_id, color)
                        self._samples_since_fit += 1
                        feat = color.astype(np.float64)
                        if feat.shape[0] == self.kmeans.cluster_centers_.shape[1]:
                            pred_cluster = int(self.kmeans.predict([feat])[0])
                            pred_label = self.cluster_role_map[pred_cluster]
                            self._record_vote_and_maybe_update(track_id, pred_label)
                            self._maybe_refit_kmeans()

                det["team"] = self.team_assignment[track_id]
                continue

            # --- 2) Henüz sabitlenmediyse bu karedeki forma betimleyicisini çıkar ve geçmişe ekle ---
            color = self.extract_jersey_color(frame, det["bbox"])
            if color is not None:
                self._collect_sample(track_id, color)
                self._samples_since_fit += 1

            # --- 3) Henüz kesin karar yoksa, model fit edilmeye hazır mı diye dene ---
            self._try_fit_kmeans()

            if track_id in self.team_assignment:
                det["team"] = self.team_assignment[track_id]
            elif self.is_fitted:
                avg_color = self._get_average_color(track_id)
                if avg_color is not None:
                    predicted_cluster = int(self.kmeans.predict([avg_color])[0])
                    label = self.cluster_role_map[predicted_cluster]
                    det["team"] = label
                    history = self.color_history.get(track_id)
                    if history is not None and len(history) >= self.min_samples_for_lock:
                        self.team_assignment[track_id] = label
                        self._store_track_feature_bank(track_id, avg_color)
                        self._vote_history[track_id] = deque([label], maxlen=self.vote_window)
                        self._track_seen_count[track_id] = 0
                        self.color_history.pop(track_id, None)
                        self._maybe_refit_kmeans()
                else:
                    det["team"] = None
            else:
                det["team"] = None

        # Aynı kare içinde K-Means ilk kez fit edildiyse, döngünün başındaki
        # oyuncuların etiketlerini de yeni dolan team_assignment'tan tamamla.
        if self.is_fitted:
            for det in outfield_dets:
                if det.get("team") is None:
                    tid = det.get("track_id")
                    if tid in self.team_assignment:
                        det["team"] = self.team_assignment[tid]

        # Takım savunma/saha içi yatay konum EMA'larını güncelle ve kalecileri çoğunluk oyuyla kilitle
        self._update_team_spatial_ema(outfield_dets)
        min_gk_agree = float(getattr(config, "TEAM_CLASSIFIER_VOTE_MIN_AGREEMENT", 0.6))
        for det in gk_dets:
            tid = det.get("track_id")
            if tid is not None and tid in self._locked_gk_tracks and tid in self.team_assignment:
                det["team"] = self.team_assignment[tid]
                continue

            matched = self._resolve_goalkeeper_team(det["bbox"], outfield_dets)
            if tid is None:
                det["team"] = matched
                continue

            if matched in ("Team A", "Team B"):
                votes = self._gk_vote_history.get(tid)
                if votes is None:
                    votes = deque(maxlen=self.vote_window)
                    self._gk_vote_history[tid] = votes
                votes.append(matched)

                if len(votes) >= self.min_samples_for_lock:
                    top_label, top_cnt = Counter(votes).most_common(1)[0]
                    if (
                        top_label in ("Team A", "Team B")
                        and (top_cnt / len(votes)) >= min_gk_agree
                        and top_cnt >= self.min_samples_for_lock
                    ):
                        self.team_assignment[tid] = top_label
                        self._locked_gk_tracks.add(tid)
                        self._gk_vote_history.pop(tid, None)
                        det["team"] = top_label
                        continue

            det["team"] = "Goalkeeper"

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
                        predicted_cluster = int(self.kmeans.predict([avg_color])[0])
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
