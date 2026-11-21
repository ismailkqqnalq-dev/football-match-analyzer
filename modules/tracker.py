"""
tracker.py
----------
ADIM 3: NESNE TAKİBİ (OBJECT TRACKING) MODÜLÜ

Bu modülün görevi: detector.py'nin ürettiği "her karede sıfırdan tespit"
listesini alıp, kareler arasında **kalıcı kimlik (ID) sürekliliği** sağlamak.

Neden bu ayrı bir problem?
    YOLO her kareyi BAĞIMSIZ olarak işler; yani 5. karedeki "oyuncu" ile
    6. karedeki "oyuncu" arasında hiçbir bağlantı kurmaz, ikisi de sıfırdan
    tespit edilir. Tracking, bu tespitleri zaman içinde eşleştirip her
    nesneye "Bu, 3. karede gördüğümüz 7 numaralı kişiyle aynı kişi" diyen
    algoritmadır.

Neden DeepSORT (ByteTrack değil)?
    ByteTrack SADECE hareket (Kalman filtresi + IOU eşleştirme) kullanır.
    Bir oyuncu kamera açısından birkaç saniyeliğine çıkıp geri girdiğinde,
    ByteTrack onun "yeni bir kişi" olduğunu düşünüp yeni bir ID verir çünkü
    hareket geçmişi kopmuştur.
    DeepSORT ise her track için bir "görünüm embedding'i" (forma rengi,
    doku, şekil gibi görsel özellikler) saklar. Oyuncu geri döndüğünde,
    konumu tutmasa bile GÖRÜNÜMÜ üzerinden eski ID'sini geri alabilir.
    Bu yüzden yayın kameralarında (oyuncuların sık sık kadraj dışına
    çıktığı senaryo) DeepSORT daha uygun bir tercih.

Top (ball) NEDEN AYRI YÖNETİLİYOR?
    DeepSORT'un görünüm modeli, "kişi" görüntüleri üzerinde eğitilmiş
    CNN'lere dayanır ve küçük/tekdüze nesneler (top gibi) için güvenilir
    embedding üretmez. Sahada zaten tek bir top olduğundan, karmaşık bir
    takip algoritmasına gerek yok: her karede en yüksek güven skorlu tek
    top tespitini alıp sabit "ball" ID'si atıyoruz, kayıp karelerde ise
    kısa süreliğine son bilinen konumu koruyoruz.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.track import TrackState

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """İki [x1, y1, x2, y2] kutu kümesi arasındaki çiftli IoU matrisini döndürür."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union > 0.0, inter_area / np.maximum(union, 1e-12), 0.0)


class _KalmanBoxFilter:
    """ByteTrack için 8 durumlu (cx, cy, a, h, vx, vy, va, vh) sabit hızlı Kalman filtresi."""

    def __init__(self):
        self._motion_mat = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0
        self._update_mat = np.eye(4, 8, dtype=np.float64)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    @staticmethod
    def xyxy_to_xyah(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.array([cx, cy, w / h, h], dtype=np.float64)

    @staticmethod
    def xyah_to_xyxy(state: np.ndarray) -> List[float]:
        cx, cy, a, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        h = max(1.0, h)
        a = max(1e-3, a)
        w = a * h
        return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = measurement
        h = max(1.0, float(measurement[3]))
        std = np.array([
            2.0 * self._std_weight_position * h,
            2.0 * self._std_weight_position * h,
            1e-2,
            2.0 * self._std_weight_position * h,
            10.0 * self._std_weight_velocity * h,
            10.0 * self._std_weight_velocity * h,
            1e-5,
            10.0 * self._std_weight_velocity * h,
        ], dtype=np.float64)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        std = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ], dtype=np.float64)
        motion_cov = np.diag(np.square(std))

        next_mean = self._motion_mat @ mean
        next_mean[2] = max(1e-3, float(next_mean[2]))
        next_mean[3] = max(1.0, float(next_mean[3]))
        next_cov = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return next_mean, next_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        std = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ], dtype=np.float64)
        innovation_cov = np.diag(np.square(std))

        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        kalman_gain = np.linalg.solve(proj_cov.T, (covariance @ self._update_mat.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_mean[2] = max(1e-3, float(new_mean[2]))
        new_mean[3] = max(1.0, float(new_mean[3]))
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


@dataclass
class _ByteTrackState:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    age: int
    time_since_update: int
    confirmed: bool
    det_class: str
    det_conf: Optional[float]
    last_det_xyxy: Optional[List[float]]


class _ByteTrackEngine:
    """Saf NumPy/SciPy tabanlı iki aşamalı IoU + Kalman takip motoru."""

    def __init__(
        self,
        max_age: int,
        n_init: int,
        max_iou_distance: float,
        high_conf_thresh: float,
        low_conf_thresh: float,
        new_track_thresh: float,
    ):
        self.max_age = max(1, int(max_age))
        self.n_init = max(1, int(n_init))
        self.max_iou_distance = float(max_iou_distance)
        self.high_conf_thresh = float(high_conf_thresh)
        self.low_conf_thresh = float(low_conf_thresh)
        self.new_track_thresh = float(new_track_thresh)

        self._kf = _KalmanBoxFilter()
        self.tracks: List[_ByteTrackState] = []
        self._next_id: int = 1

    def clear_tracks_keep_id(self) -> None:
        """Aktif track listesini temizler, kimlik çakışmasını önlemek için _next_id'yi korur."""
        self.tracks = []

    def _associate(
        self,
        track_indices: List[int],
        det_indices: List[int],
        det_boxes: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)

        t_boxes = np.array(
            [self._kf.xyah_to_xyxy(self.tracks[ti].mean) for ti in track_indices],
            dtype=np.float64,
        )
        d_boxes = det_boxes[det_indices]
        iou = _compute_iou_matrix(t_boxes, d_boxes)
        raw_cost = 1.0 - iou
        gated_cost = np.where(raw_cost <= self.max_iou_distance, raw_cost, 1e5)

        row_ind, col_ind = linear_sum_assignment(gated_cost)

        matches: List[Tuple[int, int]] = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if raw_cost[r, c] <= self.max_iou_distance:
                ti = track_indices[r]
                di = det_indices[c]
                matches.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        unmatched_tracks = [ti for ti in track_indices if ti not in matched_t]
        unmatched_dets = [di for di in det_indices if di not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, person_detections: List[Dict]) -> List[Dict]:
        # 1) Tüm mevcut track'leri 1 adım ileri tahmin et
        for t in self.tracks:
            t.mean, t.covariance = self._kf.predict(t.mean, t.covariance)
            t.age += 1
            t.time_since_update += 1
            t.det_conf = None
            t.last_det_xyxy = None

        # 2) Geçerli tespitleri yüksek/düşük güven gruplarına ayır
        valid_dets: List[Dict] = []
        valid_boxes: List[List[float]] = []
        high_indices: List[int] = []
        low_indices: List[int] = []

        for det in person_detections:
            x1, y1, x2, y2 = map(float, det["bbox"])
            if x2 <= x1 or y2 <= y1:
                continue
            idx = len(valid_dets)
            valid_dets.append(det)
            valid_boxes.append([x1, y1, x2, y2])
            conf = det.get("confidence")
            conf_val = float(conf) if conf is not None else 1.0
            if conf_val >= self.high_conf_thresh:
                high_indices.append(idx)
            elif conf_val >= self.low_conf_thresh:
                low_indices.append(idx)

        det_boxes_arr = (
            np.asarray(valid_boxes, dtype=np.float64)
            if valid_boxes
            else np.zeros((0, 4), dtype=np.float64)
        )

        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.confirmed]
        unconfirmed_indices = [i for i, t in enumerate(self.tracks) if not t.confirmed]

        # Aşama 1: Onaylı track'ler <-> yüksek güvenli tespitler
        matches_1, unm_conf_1, unm_high_1 = self._associate(
            confirmed_indices, high_indices, det_boxes_arr
        )
        # Aşama 2: Kalan onaylı track'ler <-> düşük güvenli tespitler (kapanma toparlama)
        matches_2, _, _ = self._associate(
            unm_conf_1, low_indices, det_boxes_arr
        )
        # Aşama 3: Aday (unconfirmed) track'ler <-> kalan yüksek güvenli tespitler
        matches_3, _, unm_high_final = self._associate(
            unconfirmed_indices, unm_high_1, det_boxes_arr
        )

        for ti, di in matches_1 + matches_2 + matches_3:
            t = self.tracks[ti]
            det = valid_dets[di]
            box = valid_boxes[di]
            t.mean, t.covariance = self._kf.update(
                t.mean, t.covariance, self._kf.xyxy_to_xyah(box)
            )
            t.hits += 1
            t.time_since_update = 0
            t.det_class = det.get("class_name") or t.det_class
            t.det_conf = det.get("confidence")
            t.last_det_xyxy = box
            if not t.confirmed and t.hits >= self.n_init:
                t.confirmed = True

        # Süresi dolan veya onaylanmadan kaybolan track'leri temizle
        kept_tracks: List[_ByteTrackState] = []
        for t in self.tracks:
            if t.confirmed and t.time_since_update <= self.max_age:
                kept_tracks.append(t)
            elif not t.confirmed and t.time_since_update == 0:
                kept_tracks.append(t)
        self.tracks = kept_tracks

        # Kalan yüksek güvenli tespitlerden yeni track başlat
        fallback_cls = next(iter(config.HUMAN_CLASS_NAMES), "person")
        for di in unm_high_final:
            det = valid_dets[di]
            conf = det.get("confidence")
            conf_val = float(conf) if conf is not None else 1.0
            if conf_val < self.new_track_thresh:
                continue
            box = valid_boxes[di]
            mean, cov = self._kf.initiate(self._kf.xyxy_to_xyah(box))
            self.tracks.append(
                _ByteTrackState(
                    track_id=self._next_id,
                    mean=mean,
                    covariance=cov,
                    hits=1,
                    age=1,
                    time_since_update=0,
                    confirmed=(self.n_init <= 1),
                    det_class=det.get("class_name") or fallback_cls,
                    det_conf=det.get("confidence"),
                    last_det_xyxy=box,
                )
            )
            self._next_id += 1

        # Sadece onaylanmış track'leri standart şemada döndür
        results: List[Dict] = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            bbox = (
                list(map(float, t.last_det_xyxy))
                if t.last_det_xyxy is not None
                else self._kf.xyah_to_xyxy(t.mean)
            )
            results.append({
                "bbox": bbox,
                "track_id": int(t.track_id),
                "class_name": t.det_class or fallback_cls,
                "confidence": round(float(t.det_conf), 4) if t.det_conf is not None else None,
                "team": None,
                "is_predicted": t.time_since_update > 0,
            })
        return results


class PlayerTracker:
    """
    "İnsan" sayılan (config.HUMAN_CLASS_NAMES'te tanımlı) tespitleri DeepSORT
    ile takip eden sınıf. Bu isim kümesi, kullanılan YOLO modeline göre değişir:
        - COCO pretrained model: {"person"} (oyuncu/hakem henüz ayrılmamış)
        - Fine-tune futbol modeli: {"player", "goalkeeper", "referee"}
    Bu sayede aynı PlayerTracker kodu, hangi model kullanılırsa kullanılsın
    değişiklik gerektirmeden çalışır.

    Kullanım örneği:
        tracker = PlayerTracker()
        tracked = tracker.update(detections, frame)
        # tracked -> track_id alanı doldurulmuş detection listesi
    """

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        n_init: int = config.TRACKER_N_INIT,
        max_cosine_distance: float = config.TRACKER_MAX_COSINE_DISTANCE,
        nn_budget: int = config.TRACKER_NN_BUDGET,
        embedder: str = config.TRACKER_EMBEDDER,
        embedder_gpu: bool = config.TRACKER_EMBEDDER_GPU,
        half: bool = config.TRACKER_EMBEDDER_HALF,
        tracker_type: Optional[str] = None,
        embedding_stride: Optional[int] = None,
        max_iou_distance: Optional[float] = None,
    ):
        raw_type = (
            tracker_type
            if tracker_type is not None
            else getattr(config, "TRACKER_TYPE", "deepsort")
        )
        normalized = str(raw_type).strip().lower()
        if normalized in {"deepsort", "deep_sort"}:
            self.tracker_type = "deepsort"
        elif normalized in {"bytetrack", "fast", "iou", "kalman_iou"}:
            self.tracker_type = "bytetrack"
        else:
            raise ValueError(
                f"Desteklenmeyen tracker_type='{raw_type}'. "
                f"Beklenen değerler: 'deepsort', 'bytetrack', 'fast', 'iou', 'kalman_iou'."
            )

        raw_stride = (
            embedding_stride
            if embedding_stride is not None
            else getattr(config, "TRACKER_EMBEDDING_STRIDE", 1)
        )
        self.embedding_stride = max(1, int(raw_stride))

        raw_iou = (
            max_iou_distance
            if max_iou_distance is not None
            else getattr(config, "TRACKER_MAX_IOU_DISTANCE", 0.7)
        )
        self.max_iou_distance = float(raw_iou)
        self._frame_counter: int = 0
        self.tracker = None
        self._bt_engine: Optional[_ByteTrackEngine] = None

        if self.tracker_type == "bytetrack":
            print(
                f"[PlayerTracker] ByteTrack (Kalman+IoU) başlatılıyor "
                f"(max_age={max_age}, n_init={n_init}, max_iou_distance={self.max_iou_distance})"
            )
            self._bt_engine = _ByteTrackEngine(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=self.max_iou_distance,
                high_conf_thresh=float(getattr(config, "TRACKER_HIGH_CONF_THRESH", 0.45)),
                low_conf_thresh=float(getattr(config, "TRACKER_LOW_CONF_THRESH", 0.05)),
                new_track_thresh=float(getattr(config, "TRACKER_NEW_TRACK_CONF_THRESH", 0.3)),
            )
        else:
            if embedder_gpu and not torch.cuda.is_available():
                print("[PlayerTracker] UYARI: embedder_gpu=True istendi ancak CUDA yok; CPU'ya düşülüyor.")
                embedder_gpu = False

            use_half = bool(half and embedder_gpu)

            print(
                f"[PlayerTracker] DeepSORT başlatılıyor "
                f"(max_age={max_age}, n_init={n_init}, embedder={embedder}, "
                f"embedder_gpu={embedder_gpu}, half={use_half}, "
                f"embedding_stride={self.embedding_stride})"
            )

            # DeepSort nesnesi hem Kalman filtresi (hareket tahmini) hem de
            # appearance embedder'ı (görünüm karşılaştırması) içeride yönetir.
            self.tracker = DeepSort(
                max_age=max_age,                          # ID'nin "hayatta" kalma süresi (frame)
                n_init=n_init,                            # onay için gereken ardışık eşleşme sayısı
                max_cosine_distance=max_cosine_distance,  # görünüm benzerliği eşiği
                max_iou_distance=self.max_iou_distance,
                nn_budget=nn_budget,                      # track başına saklanan embedding sayısı
                embedder=embedder,                        # "mobilenet" -> hafif CNN
                embedder_gpu=embedder_gpu,
                bgr=True,                                 # OpenCV kareleri BGR formatındadır
                half=use_half,                            # GPU'da FP16 hızlandırma
            )

    # ----------------------------------------------------------------
    def on_scene_cut(self) -> None:
        """
        TV yayını sahne kesmesinde aktif track'leri temizler; kimlik çakışmasını
        önlemek için monoton artan _next_id sayacını sıfırlamadan korur.
        """
        if self.tracker_type == "bytetrack" and self._bt_engine is not None:
            self._bt_engine.clear_tracks_keep_id()
        elif self.tracker is not None and hasattr(self.tracker, "tracker"):
            inner = self.tracker.tracker
            saved_next_id = getattr(inner, "_next_id", 1)
            inner.tracks = []
            inner.del_tracks_ids = []
            if hasattr(inner, "metric") and hasattr(inner.metric, "samples"):
                inner.metric.samples.clear()
            inner._next_id = saved_next_id
        self._frame_counter = 0

    def reset_on_cut(self) -> None:
        """on_scene_cut() için uyumluluk takma adı."""
        self.on_scene_cut()

    # ----------------------------------------------------------------
    def _update_deepsort_iou_step(
        self,
        raw_detections: List[Tuple[List[float], float, str]],
        frame: Optional[np.ndarray],
    ) -> List:
        """
        embedding_stride > 1 iken ara karelerde mevcut track'leri Kalman+IoU ile
        günceller; yalnızca eşleşmeyen yeni tespitler için appearance embedding üretir.
        """
        inner = self.tracker.tracker
        inner.predict()

        valid_raw = [d for d in raw_detections if d[0][2] > 0 and d[0][3] > 0]

        matches: List[Tuple[int, int]] = []
        unmatched_tracks: List[int] = list(range(len(inner.tracks)))
        unmatched_dets: List[int] = list(range(len(valid_raw)))

        if inner.tracks and valid_raw:
            track_boxes = []
            for t in inner.tracks:
                l, top, w, h = t.to_ltwh()
                track_boxes.append([float(l), float(top), float(l + w), float(top + h)])
            det_boxes = [
                [float(d[0][0]), float(d[0][1]), float(d[0][0] + d[0][2]), float(d[0][1] + d[0][3])]
                for d in valid_raw
            ]
            iou = _compute_iou_matrix(np.asarray(track_boxes), np.asarray(det_boxes))
            raw_cost = 1.0 - iou
            gated_cost = np.where(raw_cost <= self.max_iou_distance, raw_cost, 1e5)
            row_ind, col_ind = linear_sum_assignment(gated_cost)

            matched_t = set()
            matched_d = set()
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if raw_cost[r, c] <= self.max_iou_distance:
                    matches.append((r, c))
                    matched_t.add(r)
                    matched_d.add(c)
            unmatched_tracks = [i for i in range(len(inner.tracks)) if i not in matched_t]
            unmatched_dets = [j for j in range(len(valid_raw)) if j not in matched_d]

        for t_idx, d_idx in matches:
            track = inner.tracks[t_idx]
            ltwh, conf, cls_name = valid_raw[d_idx]
            det_obj = Detection(ltwh, conf, np.empty(0, dtype=np.float32), class_name=cls_name)
            track.original_ltwh = det_obj.get_ltwh()
            track.mean, track.covariance = inner.kf.update(
                track.mean, track.covariance, det_obj.to_xyah()
            )
            track.det_conf = det_obj.confidence
            track.det_class = det_obj.class_name
            track.hits += 1
            track.time_since_update = 0
            if track.state == TrackState.Tentative and track.hits >= track._n_init:
                track.state = TrackState.Confirmed

        for t_idx in unmatched_tracks:
            inner.tracks[t_idx].mark_missed()

        if unmatched_dets and frame is not None:
            unmatched_raw = [valid_raw[i] for i in unmatched_dets]
            with torch.inference_mode():
                embeds = self.tracker.generate_embeds(frame, unmatched_raw)
            new_dets = self.tracker.create_detections(unmatched_raw, embeds)
            for det_obj in new_dets:
                inner._initiate_track(det_obj)

        new_tracks = []
        inner.del_tracks_ids = []
        for t in inner.tracks:
            if not t.is_deleted():
                new_tracks.append(t)
            else:
                inner.del_tracks_ids.append(t.track_id)
        inner.tracks = new_tracks

        active_targets = [t.track_id for t in inner.tracks if t.is_confirmed()]
        features, targets = [], []
        samples_dict = getattr(inner.metric, "samples", {})
        for t in inner.tracks:
            if not t.is_confirmed():
                continue
            if t.track_id not in samples_dict and t.features:
                features.extend(t.features)
                targets.extend([t.track_id] * len(t.features))
                t.features = [t.features[-1]]
        inner.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

        return inner.tracks

    # ----------------------------------------------------------------
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Bir karedeki insan tespitlerini tracker'a besleyip, kalıcı track_id
        atanmış güncel detection listesini döndürür.

        ÖNEMLİ: Bu metod SADECE config.HUMAN_CLASS_NAMES içindeki class_name'e
        sahip tespitleri işler. Top, ayrı olarak BallTracker ile yönetilir (aşağıda).
        """
        # Sadece "insan" sayılan tespitleri alıyoruz (oyuncu/kaleci/hakem --
        # hangi sınıf isimlerinin buraya dahil olduğu config.HUMAN_CLASS_NAMES
        # ile TARGET_CLASS_IDS'ten otomatik türetilir, bkz. config.py).
        person_detections = [d for d in detections if d["class_name"] in config.HUMAN_CLASS_NAMES]

        if self.tracker_type == "bytetrack":
            return self._bt_engine.update(person_detections)

        # DeepSORT'un beklediği format: [([left, top, width, height], confidence, class_name), ...]
        # Bizim bbox formatımız [x1, y1, x2, y2] olduğu için width/height'a çeviriyoruz.
        raw_detections = []
        for det in person_detections:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            raw_detections.append(([x1, y1, width, height], det["confidence"], det["class_name"]))

        self._frame_counter += 1
        use_full_embed = (
            self.embedding_stride <= 1
            or (self._frame_counter - 1) % self.embedding_stride == 0
        )

        # update_tracks: hem Kalman tahminini günceller hem yeni tespitlerle
        # eşleştirme (appearance + motion) yapar hem de görünüm embedding'ini
        # frame'den çıkarır. inference_mode ile autograd grafiği birikimini önlüyoruz.
        if use_full_embed:
            with torch.inference_mode():
                if frame is None and not raw_detections:
                    tracks = self.tracker.update_tracks(raw_detections, embeds=[], frame=None)
                else:
                    tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        else:
            tracks = self._update_deepsort_iou_step(raw_detections, frame)

        tracked_detections: List[Dict] = []
        fallback_cls = next(iter(config.HUMAN_CLASS_NAMES), "person")

        for track in tracks:
            # is_confirmed(): track'in n_init eşiğini geçip "gerçek" kabul
            # edildiğini belirtir. Onaylanmamış track'leri (henüz kararsız
            # adaylar) çıktıya dahil etmiyoruz.
            if not track.is_confirmed():
                continue

            l, t, w, h = track.to_ltwh()
            x1, y1, x2, y2 = float(l), float(t), float(l + w), float(t + h)

            original_class_name = (
                (track.get_det_class() if hasattr(track, "get_det_class") else getattr(track, "det_class", None))
                or fallback_cls
            )
            det_conf = track.get_det_conf() if hasattr(track, "get_det_conf") else None

            tracked_detections.append({
                "bbox": [x1, y1, x2, y2],
                "track_id": int(track.track_id),
                "class_name": original_class_name,
                "confidence": round(float(det_conf), 4) if det_conf is not None else None,
                "team": None,        # Adım 4'te (K-Means) doldurulacak
                "is_predicted": track.time_since_update > 0,  # bu karede gerçek tespit yok, tahmin var
            })

        return tracked_detections


class BallTracker:
    """
    Topu takip eden sönümlenen hızlı (damped velocity) tekil (singleton) sınıf.

    DeepSORT gibi bir görünüm modeli kullanmaz; bunun yerine:
      1) Her karede en yüksek güven skorlu top tespitini seçer ve piksel hızını günceller.
      2) Top geçici olarak tespit edilemezse, (config.BALL_MAX_MISSING_FRAMES)
         kare boyunca son bilinen konumu sönümlenen hızla ileri taşır ("tahmini").
      3) Bu süre aşılırsa topu "kayıp" sayar ve None döndürür.
    """

    def __init__(
        self,
        max_missing_frames: int = config.BALL_MAX_MISSING_FRAMES,
        velocity_damping: float = config.BALL_VELOCITY_DAMPING,
        max_velocity_px: float = config.BALL_MAX_VELOCITY_PX_PER_FRAME,
    ):
        self.max_missing_frames = max_missing_frames
        self.velocity_damping = velocity_damping
        self.max_velocity_px = max_velocity_px
        self.last_known_detection: Optional[Dict] = None
        self.missing_frame_count: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_real_center: Optional[Tuple[float, float]] = None
        self._frames_since_real: int = 0

    # ----------------------------------------------------------------
    def reset(self) -> None:
        """Top takip durumunu sıfırlar."""
        self.last_known_detection = None
        self.missing_frame_count = 0
        self.velocity = (0.0, 0.0)
        self._last_real_center = None
        self._frames_since_real = 0

    def on_scene_cut(self) -> None:
        """Sahne kesmesinde top hız ve tahmin durumunu sıfırlar."""
        self.reset()

    def reset_on_cut(self) -> None:
        """on_scene_cut() için uyumluluk takma adı."""
        self.on_scene_cut()

    # ----------------------------------------------------------------
    @staticmethod
    def _clamp_bbox_to_frame(bbox: List[float], frame_shape: Optional[Tuple[int, ...]]) -> List[float]:
        """Verilen bbox'ı kare boyutları içine (0..w, 0..h) kelepçeler."""
        if frame_shape is None or len(frame_shape) < 2:
            return bbox
        h, w = float(frame_shape[0]), float(frame_shape[1])
        x1, y1, x2, y2 = map(float, bbox)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= w:
            x1 = max(0.0, min(w - bw, x1))
            x2 = x1 + bw
        else:
            x1, x2 = 0.0, w
        if bh <= h:
            y1 = max(0.0, min(h - bh, y1))
            y2 = y1 + bh
        else:
            y1, y2 = 0.0, h
        return [x1, y1, x2, y2]

    # ----------------------------------------------------------------
    def update(
        self,
        detections: List[Dict],
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[Dict]:
        """
        Parametreler
        ------------
        detections : List[Dict]
            O karedeki tüm ham tespitler (person + ball karışık halde).
        frame_shape : Tuple[int, ...], optional
            Tahmin edilen konumun kare sınırları dışına taşmaması için (H, W, ...).

        Dönüş
        -----
        Optional[Dict]
            Topun güncel konumunu içeren tek bir detection sözlüğü, ya da
            top uzun süredir kayıpsa None.
        """
        ball_detections = [d for d in detections if d["class_name"] == config.BALL_CLASS_NAME]

        if ball_detections:
            # Birden fazla "top" tespiti varsa (yanlış pozitif olabilir),
            # en yüksek güven skoruna sahip olanı gerçek top kabul ediyoruz.
            best_detection = dict(max(ball_detections, key=lambda d: d["confidence"]))
            x1, y1, x2, y2 = map(float, best_detection["bbox"])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self._last_real_center is not None:
                self._frames_since_real += 1
                if self._frames_since_real <= self.max_missing_frames:
                    gap = max(1, self._frames_since_real)
                    vx = (cx - self._last_real_center[0]) / gap
                    vy = (cy - self._last_real_center[1]) / gap
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    if speed > self.max_velocity_px > 0:
                        scale = self.max_velocity_px / speed
                        vx *= scale
                        vy *= scale
                    self.velocity = (vx, vy)
                else:
                    self.velocity = (0.0, 0.0)
            else:
                self.velocity = (0.0, 0.0)

            self._last_real_center = (cx, cy)
            self._frames_since_real = 0

            best_detection["track_id"] = "ball"
            best_detection["is_predicted"] = False

            self.last_known_detection = dict(best_detection)
            self.last_known_detection["bbox"] = [x1, y1, x2, y2]
            self.missing_frame_count = 0
            return best_detection

        # Bu karede top tespit edilemedi
        self.missing_frame_count += 1
        if self._last_real_center is not None:
            self._frames_since_real += 1

        if self.last_known_detection is not None and self.missing_frame_count <= self.max_missing_frames:
            vx = self.velocity[0] * self.velocity_damping
            vy = self.velocity[1] * self.velocity_damping
            self.velocity = (vx, vy)

            x1, y1, x2, y2 = map(float, self.last_known_detection["bbox"])
            shifted_bbox = self._clamp_bbox_to_frame(
                [x1 + vx, y1 + vy, x2 + vx, y2 + vy],
                frame_shape,
            )
            self.last_known_detection["bbox"] = shifted_bbox

            predicted = dict(self.last_known_detection)
            predicted["bbox"] = list(shifted_bbox)
            predicted["is_predicted"] = True
            return predicted

        # Top uzun süredir görünmüyor -> kayıp kabul et
        return None


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa Adım 1+2+3'ü uçtan uca test eder:
# Video oku -> tespit et -> takip et -> çiz -> yeni videoya yaz.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modules.video_processor import VideoProcessor
    from modules.detector import ObjectDetector

    detector = ObjectDetector()
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    with VideoProcessor(config.DEFAULT_INPUT_VIDEO, config.DEFAULT_OUTPUT_VIDEO) as vp:
        for frame_idx, frame in vp.read_frames():
            # 1) Ham tespitler (Adım 2)
            raw_detections = detector.detect_frame(frame)

            # 2) Oyuncu/hakem takibi (DeepSORT, kalıcı ID)
            tracked_players = player_tracker.update(raw_detections, frame)

            # 3) Top takibi (basit singleton mantık)
            tracked_ball = ball_tracker.update(raw_detections)

            # 4) İkisini birleştirip çizim için tek listeye koy
            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            # 5) Çiz ve kaydet
            annotated_frame = vp.draw_detections(frame, all_tracked)
            vp.write_frame(annotated_frame)

    print("Takip testi tamamlandı. Her oyuncunun artık kalıcı bir ID'si olmalı.")
