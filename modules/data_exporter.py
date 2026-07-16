"""
data_exporter.py
------------------
ADIM 6: VERİ DIŞA AKTARIM (Pandas / CSV / SQL)

AMAÇ: Pipeline'ın önceki adımlarında (2-5) her karede ürettiğimiz zengin bilgiyi
(kim nerede, hangi takımda, saha üzerinde tam olarak hangi metrede) kalıcı,
analiz edilebilir bir forma dönüştürmek. Video bittiğinde elimizde sadece
"annotated video" değil, Excel'de açılabilen bir CSV ya da SQL sorgusu
yazılabilen bir veritabanı da olmalı -- bir maç analistinin asıl ihtiyacı
budur (video zaten göz ile izlenebilir, ama "Oyuncu 7 maçta toplam kaç metre
koştu?" gibi bir soruya video izleyerek cevap verilemez).

TASARIM: DataExporter bir "biriktirici" (accumulator) gibi çalışır.
    1) main.py video döngüsünde HER KAREDE  exporter.append_frame_data(...) çağırır
       -> bellekte bir liste (self.records) büyür, her eleman TEK BİR
       tespitin (oyuncu ya da top) o karedeki tüm bilgisini taşıyan bir satır.
    2) Video bitince exporter.export_to_csv() / export_to_sql() çağrılır
       -> bu liste bir pandas DataFrame'e çevrilip diske yazılır.
    3) İsteğe bağlı: exporter.export_distance_summary() -> saha koordinatlarını
       kullanarak oyuncu başına "toplam kat edilen mesafe" ve "ortalama/maksimum
       hız" özet tablosu üretir (Adım 5'teki homografi çalışmasının doğrudan
       meyvesi -- piksel koordinatlarıyla bu hesap ANLAMSIZ olurdu, çünkü
       kameraya yakın/uzak oyuncular farklı piksel/metre oranına sahiptir).

NEDEN BELLEKTE BİRİKTİRİP SONDA YAZIYORUZ (her karede diske yazmak yerine)?
    Disk I/O, her karede yapılırsa (bir maçta 100.000+ kare olabilir) ciddi
    yavaşlığa yol açar. Bunun yerine hafif Python sözlükleri bellekte
    tutulur, video biter bitmez TEK SEFERDE pandas DataFrame'e ve diske
    yazılır. Çok uzun maçlarda (bellek endişesi varsa) bu yaklaşım periyodik
    "flush" (ör. her 10.000 karede bir ara CSV'ye ekle) şeklinde
    genişletilebilir -- şimdilik kapsam dışı bırakıldı.
"""

import os
import sqlite3
import sys
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.perspective_transformer import PerspectiveTransformer


class DataExporter:
    """
    Kare-kare takip verisini biriktirip CSV/SQL'e yazan ve saha koordinatları
    üzerinden oyuncu mesafe/hız özeti çıkaran sınıf.
    """

    def __init__(self, fps: Optional[float] = None):
        """
        Parametreler
        ------------
        fps : float, optional
            Videonun saniyedeki kare sayısı (VideoProcessor.fps). Verilirse,
            her satıra "timestamp_sec" (frame_idx / fps) eklenir -- bu,
            zaman bazlı analiz/senkronizasyon için kullanışlıdır. Verilmezse
            bu alan None bırakılır.
        """
        self.fps = fps
        self.records: List[Dict] = []

    # ----------------------------------------------------------------
    def append_frame_data(self, frame_idx: int, all_tracked: List[Dict]) -> None:
        """
        Bir karedeki tüm tespitleri (oyuncular + top) iç listeye ekler.

        Parametreler
        ------------
        frame_idx : int
            Kare numarası (main.py'deki vp.read_frames()'ten gelir).
        all_tracked : List[Dict]
            O karedeki tüm takip edilen nesneler -- her biri en azından
            "bbox", "track_id", "class_name" alanlarını içermeli. "team",
            "confidence", "is_predicted", "pitch_x", "pitch_y", "on_pitch"
            alanları varsa (Adım 3-5 pipeline'ından geçtiyse) onlar da
            kaydedilir; yoksa None olarak yazılır (esnek/geriye dönük uyumlu).
        """
        timestamp_sec = (frame_idx / self.fps) if self.fps else None

        for det in all_tracked:
            x1, y1, x2, y2 = det["bbox"]

            # "Ayak noktası" -- perspective_transformer.py'nin homografi için
            # kullandığı piksel noktasıyla BİREBİR TUTARLI olması için aynı
            # statik metodu kullanıyoruz. Böylece dışa aktarılan pixel_x/y
            # alanı, pitch_x/y'nin türetildiği kaynak noktayla eşleşir.
            foot_x, foot_y = PerspectiveTransformer.get_foot_position([x1, y1, x2, y2])

            self.records.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3) if timestamp_sec is not None else None,
                "track_id": det.get("track_id"),
                "class_name": det.get("class_name"),
                "team": det.get("team"),
                "confidence": det.get("confidence"),
                "is_predicted": det.get("is_predicted"),
                "bbox_x1": round(x1, 1),
                "bbox_y1": round(y1, 1),
                "bbox_x2": round(x2, 1),
                "bbox_y2": round(y2, 1),
                "pixel_x": round(foot_x, 1),
                "pixel_y": round(foot_y, 1),
                "pitch_x": det.get("pitch_x"),
                "pitch_y": det.get("pitch_y"),
                "on_pitch": det.get("on_pitch"),
            })

    # ----------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Biriken tüm kayıtları bir pandas DataFrame olarak döndürür."""
        return pd.DataFrame(self.records)

    # ----------------------------------------------------------------
    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Ham kare-kare takip verisini CSV olarak diske yazar.

        Dönüş: yazılan dosyanın tam yolu.
        """
        if output_path is None:
            os.makedirs(config.EXPORT_DIR, exist_ok=True)
            output_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)

        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"[DataExporter] {len(df)} satır CSV'ye yazıldı: {output_path}")
        return output_path

    # ----------------------------------------------------------------
    def export_to_sql(self, db_path: Optional[str] = None, table_name: Optional[str] = None) -> str:
        """
        Ham kare-kare takip verisini bir SQLite veritabanına yazar.
        SQLite seçildi çünkü tek dosyalık, sunucu kurulumu gerektirmeyen,
        her yerde (DB Browser, pandas, herhangi bir SQL istemcisi) doğrudan
        açılabilen bir format -- bu, projenin taşınabilirliğini bozmuyor.

        Dönüş: yazılan veritabanı dosyasının tam yolu.
        """
        if db_path is None:
            os.makedirs(config.EXPORT_DIR, exist_ok=True)
            db_path = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)
        if table_name is None:
            table_name = config.EXPORT_SQL_TABLE_NAME

        df = self.to_dataframe()

        conn = sqlite3.connect(db_path)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            # Sık kullanılacak sorguları hızlandırmak için basit indeksler.
            # (ör. "belirli bir track_id'nin tüm karelerini getir" ya da
            # "belirli bir kareyi getir" sorguları çok sık kullanılır.)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_track_id ON {table_name}(track_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_frame_idx ON {table_name}(frame_idx)")
            conn.commit()
        finally:
            conn.close()

        print(f"[DataExporter] {len(df)} satır SQLite veritabanına yazıldı: {db_path} (tablo: {table_name})")
        return db_path

    # ----------------------------------------------------------------
    def export_distance_summary(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Saha (metre) koordinatlarını kullanarak HER OYUNCU için:
            - toplam kat edilen mesafe (metre)
            - ortalama hız (km/saat)
            - maksimum ANLIK hız (km/saat, iki ardışık kare arası)
        hesaplayıp bir özet CSV'sine yazar.

        NEDEN SADECE ARDIŞIK KARELER KULLANILIYOR:
        Bir oyuncu birkaç kare kaybolup (ör. başka bir oyuncunun arkasında
        kalıp) sonra tekrar tespit edilirse, o iki nokta arasındaki "mesafe"
        gerçek koşu mesafesi DEĞİL, düz bir çizgidir ve kare sayısı azdır --
        bu da sahte, çok yüksek bir "hız" hesaplanmasına yol açar. Bu yüzden
        sadece frame_idx farkı config.DISTANCE_CALC_MAX_FRAME_GAP değerine
        eşit ya da altında olan ardışık satır çiftleri hesaba katılır.
        Benzer şekilde, Kalman filtresinin TAHMİN ettiği (gerçek tespit
        olmayan) konumlar da varsayılan ayarla hesaba katılmaz.

        Dönüş: yazılan dosyanın tam yolu, ya da hesaplanacak veri yoksa None.
        """
        df = self.to_dataframe()

        if df.empty:
            print("[DataExporter] Özet çıkarılacak veri yok (records boş).")
            return None

        # Sadece oyuncuları al (topun "mesafe koştuğu" kavramsal olarak
        # anlamlı değil / farklı bir analiz konusu -- top hızını istersen
        # ayrı bir metodla ekleyebiliriz).
        players_df = df[df["class_name"] == "person"].copy()

        # Saha koordinatı olmayan (kalibrasyon yapılmamış / homografi
        # üretilememiş) satırları ayıkla.
        players_df = players_df.dropna(subset=["pitch_x", "pitch_y"])

        if config.DISTANCE_CALC_EXCLUDE_PREDICTED:
            players_df = players_df[players_df["is_predicted"] != True]  # noqa: E712

        # on_pitch açıkça False olanları ayıkla (None ise -- kalibrasyon
        # bilgisi eksik demektir, zaten yukarıda dropna ile elenmiş olur).
        players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        if players_df.empty:
            print("[DataExporter] Mesafe hesabı için yeterli saha-koordinatlı veri yok.")
            return None

        players_df = players_df.sort_values(["track_id", "frame_idx"])

        summary_rows = []
        for track_id, group in players_df.groupby("track_id"):
            group = group.reset_index(drop=True)

            # Ardışık satırlar arası fark
            frame_diff = group["frame_idx"].diff()
            dx = group["pitch_x"].diff()
            dy = group["pitch_y"].diff()
            dist_segment = (dx ** 2 + dy ** 2) ** 0.5

            # Sadece kare farkı izin verilen aralıkta olan (yani ardışık,
            # arada kayıp kare olmayan) segmentleri say.
            valid_segment = frame_diff <= config.DISTANCE_CALC_MAX_FRAME_GAP

            total_distance_m = dist_segment[valid_segment].sum()

            # Anlık hız (km/saat) hesabı: sadece fps biliniyorsa mümkün.
            max_speed_kmh = None
            avg_speed_kmh = None
            if self.fps:
                dt_seconds = frame_diff[valid_segment] / self.fps
                speeds_ms = dist_segment[valid_segment] / dt_seconds  # m/s
                speeds_kmh = speeds_ms * 3.6
                speeds_kmh = speeds_kmh.replace([float("inf"), -float("inf")], None).dropna()
                if len(speeds_kmh) > 0:
                    max_speed_kmh = round(speeds_kmh.max(), 2)
                    avg_speed_kmh = round(speeds_kmh.mean(), 2)

            team = group["team"].dropna().iloc[0] if group["team"].notna().any() else None

            summary_rows.append({
                "track_id": track_id,
                "team": team,
                "frames_tracked": len(group),
                "total_distance_m": round(total_distance_m, 1),
                "avg_speed_kmh": avg_speed_kmh,
                "max_speed_kmh": max_speed_kmh,
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("total_distance_m", ascending=False)

        if output_path is None:
            os.makedirs(config.EXPORT_DIR, exist_ok=True)
            output_path = os.path.join(config.EXPORT_DIR, config.EXPORT_DISTANCE_SUMMARY_FILENAME)

        summary_df.to_csv(output_path, index=False)
        print(f"[DataExporter] {len(summary_df)} oyuncu için mesafe özeti yazıldı: {output_path}")
        return output_path
