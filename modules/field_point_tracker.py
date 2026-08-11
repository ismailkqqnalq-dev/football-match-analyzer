"""
field_point_tracker.py
--------------------
ADIM 7 (İLERİ SEVİYE): FUTBOL SAHASI KÖŞE NOKTALARININ KALICI İZLENMESİ

Bu modülün görevi:
- Futbol sahası köşelerini (4 ana köşe, ceza sahası köşeleri, orta daire merkezi, vb)
  VIDEO BAŞINDA TEK BİR KERE OTOMATIK olarak tespit et.
- Tespit edilen noktaları SABİT KAL --> Sahne kesintileri (scene cuts) sırasında
  noktalar KAYMASIN / SÜRÜKLENMESINN (kullanıcı için daha istikrarlı bir harita).
- Sahne kesintilerini DETECT ET (histogram farkı + optical flow anomalileri)
  ve algılanan kesintilerin ardından noktaları YENİDEN TESPİT ET.
- Tüm istikrarlı noktaları JSON ve binary (pickle) olarak kaydet.

FIELD LANDMARK TÜRLERİ (standart futbol sahası için):
  - "corner"        : 4 köşe bayrağı (sahanın 4 köşesi)
  - "goal_line"     : 2 gol çizgisi uç noktası
  - "penalty_corner": 4 ceza sahası köşesi
  - "center_line"   : 1 orta çizgi ile yan çizgisinin kesiştiği 2 nokta
  - "center_circle" : 1 orta daire merkezi (top başlangıç noktası)

OPTİMİZASYON:
- Köşe tespiti için Harris corner detection + RANSAC katılığı (robustness)
- Canny edge detection + Hough line detection ile saha çizgileri bulunur
- Çizgiler kesişerek köşe noktalarını verir

SAHNE KESİNTİSİ ALGILAMA:
- Frame farkı histogram karşılaştırması (cv2.compareHist)
- Threshold aşılırsa sahne kesintisi kabul edilir
- Kesinti algılandığında, noktalar o yeni sahnede YENİDEN TESPİT EDİLİR
"""

import cv2
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class FieldPoint:
    """Bir futbol sahası noktasını temsil eder."""
    point_id: str              # Örn: "corner_1", "penalty_box_tl"
    point_type: str            # "corner", "goal_line", "penalty_corner", "center_circle", vb
    pixel_x: float
    pixel_y: float
    pitch_x: Optional[float] = None  # Homografi ile dönüştürüldüğünde metre koordinatı
    pitch_y: Optional[float] = None
    confidence: float = 1.0    # Harris corner detection güveni (0-1)
    frame_detected: int = 0    # Hangi karede tespit edildiği
    is_stable: bool = True     # Sahne kesintisi sonrası yeniden tespit edildiyse False -> True


class FieldPointTracker:
    """
    Futbol sahası noktalarını otomatik olarak tespit eder ve kalıcı hale getirir.
    """

    # Standart futbol sahası ölçüleri (metre cinsinden) - UEFA standardı
    PITCH_LENGTH = 105.0   # metre
    PITCH_WIDTH = 68.0     # metre
    GOAL_AREA_LENGTH = 16.5  # ceza sahası uzunluğu
    GOAL_AREA_WIDTH = 40.32  # ceza sahası genişliği
    CENTER_CIRCLE_RADIUS = 9.15

    def __init__(self, use_line_detection: bool = True):
        """
        Args:
            use_line_detection: Saha çizgilerini detectlemek için Canny+Hough kullan
        """
        self.use_line_detection = use_line_detection
        self.field_points: Dict[str, FieldPoint] = {}
        self.prev_frame_gray = None
        self.prev_histogram = None
        self.scene_cut_threshold = 0.3  # Histogram CV_COMP_BHATTACHARYYA * 100 > bu eşik = scene cut
        self.detections_per_frame = []  # Tüm frame'lerdeki tespitleri sakla
        
    def detect_field_points_in_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict[str, FieldPoint]:
        """
        Verilen frame'deki futbol sahası köşe noktalarını otomatik olarak tespit et.
        
        Args:
            frame: OpenCV frame (BGR, dtype=uint8)
            frame_idx: Frame numarası (log amaçlı)
            
        Returns:
            Dictionary: {point_id: FieldPoint} formatında noktalar
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1) Harris corner detection ile köşeleri bul
        corners = self._detect_corners_harris(gray)
        
        # 2) Saha çizgileri kullarak ek nokta tespiti (isteğe bağlı)
        if self.use_line_detection:
            line_points = self._detect_field_lines(gray)
            corners.extend(line_points)
        
        # 3) Tespit edilen köşeleri düzenle ve etiketle
        field_points = self._classify_field_points(corners, frame)
        
        # 4) Bu frame'in tespitini sakla
        self.detections_per_frame.append({
            'frame_idx': frame_idx,
            'points': field_points,
            'timestamp': frame_idx  # video frame index as timestamp
        })
        
        return field_points

    def _detect_corners_harris(self, gray: np.ndarray, blockSize: int = 2, 
                                ksize: int = 3, k: float = 0.04) -> List[Tuple[float, float, float]]:
        """
        Harris corner detection kullanarak köşeleri bul.
        
        Returns:
            List of (x, y, response) tuples, sorted by response (güven) descending
        """
        # Harris corner response hesapla
        corners_response = cv2.cornerHarris(gray, blockSize, ksize, k)
        corners_response = cv2.dilate(corners_response, None)
        
        # Eşikle (güçlü köşeleri tut)
        threshold = 0.01 * corners_response.max()
        y, x = np.where(corners_response >= threshold)
        
        # (x, y, response) tuple'ları oluştur
        corners = [(x[i], y[i], corners_response[y[i], x[i]]) for i in range(len(x))]
        
        # Response'a göre azalan sırada sırala (en güçlü köşeler önce)
        corners = sorted(corners, key=lambda c: c[2], reverse=True)
        
        # En iyi N köşeyi tut (kaynakça: futbol sahası ~10-20 ana köşe)
        return corners[:20]

    def _detect_field_lines(self, gray: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Canny edge detection + Hough line detection kullanarak saha çizgilerini bul.
        Çizgilerin kesişme noktaları = köşe adayları.
        
        Returns:
            List of (x, y, confidence) tuples (çizgi kesişimleri)
        """
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return []
        
        lines = lines.reshape(-1, 4)
        intersection_points = []
        
        # Tüm çizgi çiftlerinin kesişimlerini bulabilir mi?
        # Basitleştirme: en uzun çizgileri seç (saha çizgileri genellikle uzundur)
        line_lengths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in lines]
        top_line_indices = sorted(range(len(lines)), key=lambda i: line_lengths[i], reverse=True)[:10]
        
        for i in top_line_indices:
            for j in top_line_indices:
                if i >= j:
                    continue
                # İki çizginin kesişim noktasını hesapla
                pt = self._line_intersection(lines[i], lines[j])
                if pt is not None:
                    x, y = pt
                    # Görüntü sınırları içinde mi?
                    if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                        intersection_points.append((x, y, 0.7))  # confidence=0.7
        
        return intersection_points

    @staticmethod
    def _line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        İki Hough line'ın kesişim noktasını hesapla.
        line1, line2: [x1, y1, x2, y2] formatında
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Paralel çizgiler
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return (px, py)

    def _classify_field_points(self, corners: List[Tuple[float, float, float]], 
                               frame: np.ndarray) -> Dict[str, FieldPoint]:
        """
        Tespit edilen köşeleri futbol sahası noktaları olarak sınıflandır.
        
        Heuristic yaklaşım:
        - Frame genişliği/yüksekliğini kullanarak saha öğelerinin yaklaşık konumlarını tahmin et
        - Tespit edilen köşeleri en yakın saha noktasına ata
        """
        h, w = frame.shape[:2]
        field_points = {}
        
        # Sahanın tahmini piksel konumları (video frame'e uyarlanmış)
        # NOT: Bunlar tahmini; gerçek konum kalibrasyon verisinden gelmeli
        estimated_landmarks = {
            "corner_tl": (w * 0.05, h * 0.1),       # top-left
            "corner_tr": (w * 0.95, h * 0.1),       # top-right
            "corner_br": (w * 0.95, h * 0.9),       # bottom-right
            "corner_bl": (w * 0.05, h * 0.9),       # bottom-left
            
            "goal_line_left": (w * 0.05, h * 0.5),   # goal line center
            "goal_line_right": (w * 0.95, h * 0.5),  # goal line center opposite
            
            "penalty_tl": (w * 0.2, h * 0.3),       # top-left penalty corner
            "penalty_tr": (w * 0.2, h * 0.7),       # top-right penalty corner
            "penalty_bl": (w * 0.8, h * 0.3),       # bottom-left penalty corner
            "penalty_br": (w * 0.8, h * 0.7),       # bottom-right penalty corner
            
            "center_circle": (w * 0.5, h * 0.5),    # midfield center
            "center_left": (w * 0.05, h * 0.5),     # center line left
            "center_right": (w * 0.95, h * 0.5),    # center line right
        }
        
        # Tespit edilen her köşeyi en yakın tahmini noktaya ata
        assigned = set()
        for corner_x, corner_y, response in corners:
            if len(assigned) >= len(estimated_landmarks):
                break  # Tüm noktaları atadık
            
            # En yakın tahmini noktayı bul
            min_dist = float('inf')
            best_key = None
            for key, (est_x, est_y) in estimated_landmarks.items():
                if key in assigned:
                    continue
                dist = np.sqrt((corner_x - est_x)**2 + (corner_y - est_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_key = key
            
            if best_key is not None:
                # Makul mesafe sınırı içinde mi? (çok fazla sapan noktaları reddet)
                if min_dist < max(w, h) * 0.15:  # 15% sapma toleransı
                    assigned.add(best_key)
                    point_type = best_key.split('_')[0]  # "corner", "penalty", vb
                    field_points[best_key] = FieldPoint(
                        point_id=best_key,
                        point_type=point_type,
                        pixel_x=float(corner_x),
                        pixel_y=float(corner_y),
                        confidence=float(min(1.0, response / 0.01)),  # normalize
                        frame_detected=0,
                        is_stable=True
                    )
        
        return field_points

    def detect_scene_cut(self, frame: np.ndarray) -> bool:
        """
        Sahne kesintisini (scene cut) histogram karşılaştırması ile algıla.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            bool: Scene cut algılandıysa True
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        
        if self.prev_histogram is None:
            self.prev_histogram = histogram
            return False
        
        # Histogram karşılaştırması (Bhattacharyya distance)
        distance = cv2.compareHist(self.prev_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
        
        self.prev_histogram = histogram
        
        # Eşik aşılırsa scene cut
        is_cut = distance > self.scene_cut_threshold
        
        return is_cut

    def redetect_points_on_scene_cut(self, frame: np.ndarray, frame_idx: int) -> Dict[str, FieldPoint]:
        """
        Sahne kesintisi algılandığında, noktaları o yeni sahnede YENİDEN TESPİT ET.
        Bu, noktaların kayması (drift) veya sürüklenmesini (jitter) önler.
        """
        print(f"[Scene Cut] Frame {frame_idx}'de sahne kesintisi algılandı. "
              f"Noktalar yeniden tespit ediliyor...")
        
        new_points = self.detect_field_points_in_frame(frame, frame_idx)
        
        # Yeni tespitleri mevcut field_points ile birleştir
        for point_id, point in new_points.items():
            point.is_stable = False  # Yeni tespit
            self.field_points[point_id] = point
        
        return new_points

    def update_with_homography(self, transformer) -> None:
        """
        Mevcut tüm noktaları perspektif transformeri (homografi) kullanarak
        saha koordinatlarına dönüştür.
        
        Args:
            transformer: PerspectiveTransformer örneği
        """
        for point_id, point in self.field_points.items():
            pitch_coords = transformer.pixel_to_pitch(point.pixel_x, point.pixel_y)
            if pitch_coords is not None:
                point.pitch_x, point.pitch_y = pitch_coords

    def export_to_json(self, output_path: str) -> None:
        """Tüm noktaları JSON formatında kaydet."""
        data = {
            'field_points': {},
            'metadata': {
                'pitch_length_m': self.PITCH_LENGTH,
                'pitch_width_m': self.PITCH_WIDTH,
                'total_detections': len(self.detections_per_frame)
            }
        }
        
        for point_id, point in self.field_points.items():
            data['field_points'][point_id] = asdict(point)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Export] Saha noktaları JSON'a kaydedildi: {output_path}")

    def export_to_pickle(self, output_path: str) -> None:
        """Tüm noktaları binary (pickle) formatında kaydet."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.field_points, f)
        
        print(f"[Export] Saha noktaları pickle'a kaydedildi: {output_path}")

    def export_detection_history(self, output_path: str) -> None:
        """Tüm frame'lerdeki tespit geçmişini JSON'a kaydet (debug amaçlı)."""
        data = {
            'detections_per_frame': []
        }
        
        for detection in self.detections_per_frame:
            frame_data = {
                'frame_idx': detection['frame_idx'],
                'timestamp': detection['timestamp'],
                'points_count': len(detection['points']),
                'points': {}
            }
            for point_id, point in detection['points'].items():
                frame_data['points'][point_id] = asdict(point)
            data['detections_per_frame'].append(frame_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Export] Tespit geçmişi JSON'a kaydedildi: {output_path}")

    def draw_field_points(self, frame: np.ndarray, 
                         points: Optional[Dict[str, FieldPoint]] = None) -> np.ndarray:
        """
        Saha noktalarını frame üzerine çiz.
        
        Args:
            frame: OpenCV frame
            points: Çizilecek noktalar (None ise self.field_points kullanılır)
            
        Returns:
            Annotated frame
        """
        if points is None:
            points = self.field_points
        
        # Nokta türüne göre renk ata
        color_map = {
            'corner': (0, 255, 0),         # Yeşil
            'goal_line': (255, 0, 0),      # Mavi
            'penalty': (0, 255, 255),      # Sarı
            'center': (255, 0, 255),       # Magenta
            'penalty_corner': (0, 165, 255)  # Turuncu
        }
        
        annotated = frame.copy()
        
        for point_id, point in points.items():
            # Renk seç
            color = (255, 255, 255)  # Varsayılan beyaz
            for key, col in color_map.items():
                if key in point.point_type.lower():
                    color = col
                    break
            
            # Nokta çiz (daire)
            x, y = int(point.pixel_x), int(point.pixel_y)
            cv2.circle(annotated, (x, y), 8, color, -1)
            cv2.circle(annotated, (x, y), 8, (255, 255, 255), 2)
            
            # Etiket çiz
            cv2.putText(
                annotated, point_id, (x + 10, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return annotated
