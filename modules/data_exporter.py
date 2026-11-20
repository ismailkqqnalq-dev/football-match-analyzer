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
from collections import Counter
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
    def backfill_teams(self, team_map: Optional[Dict] = None) -> int:
        """
        K-Means fit edilmeden önceki ilk karelerde `team=None` kalan oyuncu
        satırlarını, sonradan kesinleşen `track_id -> team` eşlemesiyle
        geriye dönük doldurur.

        Dönüş: doldurulan satır sayısı.
        """
        if not self.records:
            return 0

        # Kayıtlardan her track_id için en sık görülen geçerli takımı çıkar
        counts_by_tid: Dict[str, Counter] = {}
        for rec in self.records:
            if rec.get("class_name") == config.BALL_CLASS_NAME:
                continue
            team = rec.get("team")
            if team is not None and not pd.isna(team):
                tid_key = str(rec.get("track_id"))
                counts_by_tid.setdefault(tid_key, Counter())[team] += 1

        lookup: Dict[str, str] = {
            tid: counter.most_common(1)[0][0]
            for tid, counter in counts_by_tid.items()
        }

        if team_map:
            for tid, team in team_map.items():
                if team is not None and not pd.isna(team):
                    lookup[str(tid)] = team

        filled = 0
        for rec in self.records:
            if rec.get("class_name") == config.BALL_CLASS_NAME:
                continue
            team = rec.get("team")
            if team is None or pd.isna(team):
                resolved = lookup.get(str(rec.get("track_id")))
                if resolved is not None:
                    rec["team"] = resolved
                    filled += 1

        return filled

    # ----------------------------------------------------------------
    def interpolate_ball_positions(self, max_gap: int = config.BALL_INTERPOLATE_MAX_GAP) -> int:
        """
        Gerçek top tespitleri (is_predicted=False) arasındaki kısa boşlukları
        doğrusal olarak interpole eder; mevcut tahmin kayıtlarını günceller veya
        eksik ara kareler için yeni tahmin kayıtları ekler.

        Dönüş: interpole edilen kare sayısı.
        """
        if not self.records or max_gap < 2:
            return 0

        ball_by_frame: Dict[int, Dict] = {}
        real_by_frame: Dict[int, Dict] = {}
        for rec in self.records:
            if rec.get("class_name") != config.BALL_CLASS_NAME:
                continue
            f_idx = int(rec["frame_idx"])
            if rec.get("is_predicted") is False:
                real_by_frame[f_idx] = rec
                ball_by_frame[f_idx] = rec
            elif f_idx not in ball_by_frame:
                ball_by_frame[f_idx] = rec

        if len(real_by_frame) < 2:
            return 0

        real_frames = sorted(real_by_frame.keys())
        interpolated_count = 0

        for i in range(len(real_frames) - 1):
            f1 = real_frames[i]
            f2 = real_frames[i + 1]
            gap = f2 - f1
            if not (1 < gap <= max_gap):
                continue

            r1 = real_by_frame[f1]
            r2 = real_by_frame[f2]

            has_pitch = (
                r1.get("pitch_x") is not None and not pd.isna(r1.get("pitch_x"))
                and r1.get("pitch_y") is not None and not pd.isna(r1.get("pitch_y"))
                and r2.get("pitch_x") is not None and not pd.isna(r2.get("pitch_x"))
                and r2.get("pitch_y") is not None and not pd.isna(r2.get("pitch_y"))
            )

            for f in range(f1 + 1, f2):
                alpha = (f - f1) / gap
                bx1 = round(float(r1["bbox_x1"]) + alpha * (float(r2["bbox_x1"]) - float(r1["bbox_x1"])), 1)
                by1 = round(float(r1["bbox_y1"]) + alpha * (float(r2["bbox_y1"]) - float(r1["bbox_y1"])), 1)
                bx2 = round(float(r1["bbox_x2"]) + alpha * (float(r2["bbox_x2"]) - float(r1["bbox_x2"])), 1)
                by2 = round(float(r1["bbox_y2"]) + alpha * (float(r2["bbox_y2"]) - float(r1["bbox_y2"])), 1)
                foot_x, foot_y = PerspectiveTransformer.get_foot_position([bx1, by1, bx2, by2])
                pixel_x = round(foot_x, 1)
                pixel_y = round(foot_y, 1)

                if has_pitch:
                    px = round(float(r1["pitch_x"]) + alpha * (float(r2["pitch_x"]) - float(r1["pitch_x"])), 2)
                    py = round(float(r1["pitch_y"]) + alpha * (float(r2["pitch_y"]) - float(r1["pitch_y"])), 2)
                    margin = 1.0
                    on_pitch = (
                        -margin <= px <= config.PITCH_LENGTH_M + margin
                        and -margin <= py <= config.PITCH_WIDTH_M + margin
                    )
                else:
                    px, py, on_pitch = None, None, None

                ts = round(f / self.fps, 3) if self.fps else None

                if f in ball_by_frame:
                    target = ball_by_frame[f]
                    if target.get("is_predicted") is False:
                        continue
                    target["timestamp_sec"] = ts
                    target["confidence"] = None
                    target["is_predicted"] = True
                    target["bbox_x1"] = bx1
                    target["bbox_y1"] = by1
                    target["bbox_x2"] = bx2
                    target["bbox_y2"] = by2
                    target["pixel_x"] = pixel_x
                    target["pixel_y"] = pixel_y
                    target["pitch_x"] = px
                    target["pitch_y"] = py
                    target["on_pitch"] = on_pitch
                    interpolated_count += 1
                else:
                    new_rec = {
                        "frame_idx": f,
                        "timestamp_sec": ts,
                        "track_id": r1.get("track_id", "ball"),
                        "class_name": config.BALL_CLASS_NAME,
                        "team": None,
                        "confidence": None,
                        "is_predicted": True,
                        "bbox_x1": bx1,
                        "bbox_y1": by1,
                        "bbox_x2": bx2,
                        "bbox_y2": by2,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "pitch_x": px,
                        "pitch_y": py,
                        "on_pitch": on_pitch,
                    }
                    self.records.append(new_rec)
                    ball_by_frame[f] = new_rec
                    interpolated_count += 1

        self.records.sort(key=lambda r: (int(r["frame_idx"]), str(r.get("track_id"))))
        return interpolated_count

    # ----------------------------------------------------------------
    def postprocess(self, team_map: Optional[Dict] = None) -> Dict[str, int]:
        """
        Video işleme bittikten sonra veriyi temizler: ilk karelerdeki eksik
        takım etiketlerini geriye dönük doldurur ve top boşluklarını interpole eder.
        """
        backfilled = (
            self.backfill_teams(team_map=team_map)
            if config.TEAM_CLASSIFIER_BACKFILL_TEAMS
            else 0
        )
        interpolated = self.interpolate_ball_positions()
        print(
            f"[DataExporter] Postprocess tamamlandı: {backfilled} takım etiketi "
            f"dolduruldu, {interpolated} top karesi interpole edildi."
        )
        return {
            "backfilled_teams": backfilled,
            "interpolated_ball_frames": interpolated,
        }

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
            - toplam kat edilen mesafe (metre, yumuşatılmış + hız filtreli)
            - aktif takip süresi (saniye)
            - ortalama ve maksimum hız (km/saat)
            - FIFA hız bantlarına göre mesafeler (walking/jogging/running/HSR/sprint)
            - toplam sprint sayısı
        hesaplayıp bir özet CSV'sine yazar.

        Dönüş: yazılan dosyanın tam yolu, ya da hesaplanacak veri yoksa None.
        """
        df = self.to_dataframe()

        if df.empty:
            print("[DataExporter] Özet çıkarılacak veri yok (records boş).")
            return None

        # Tüm insan sınıflarını kapsa (COCO: {"person"}, fine-tune: {"player", "goalkeeper", "referee"})
        players_df = df[df["class_name"].isin(config.HUMAN_CLASS_NAMES)].copy()

        # Saha koordinatı olmayan (kalibrasyon yapılmamış / homografi
        # üretilememiş) satırları ayıkla.
        players_df = players_df.dropna(subset=["pitch_x", "pitch_y"])

        if config.DISTANCE_CALC_EXCLUDE_PREDICTED:
            players_df = players_df[players_df["is_predicted"] != True]  # noqa: E712

        # on_pitch açıkça False olanları ayıkla.
        players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        if players_df.empty:
            print("[DataExporter] Mesafe hesabı için yeterli saha-koordinatlı veri yok.")
            return None

        players_df = players_df.sort_values(["track_id", "frame_idx"])

        smooth_win = max(1, int(config.SPEED_SMOOTHING_WINDOW_FRAMES))
        min_sprint_frames = (
            max(2, int(round(self.fps * config.SPRINT_MIN_DURATION_SEC)))
            if self.fps
            else 2
        )

        summary_rows = []
        for track_id, group in players_df.groupby("track_id"):
            group = group.sort_values("frame_idx").drop_duplicates(subset=["frame_idx"]).reset_index(drop=True)

            # Sürekli segmentlere böl (arada kayıp kare boşluğu varsa ayrı segment olur)
            frame_diff = group["frame_idx"].diff()
            segment_ids = (frame_diff.isna() | (frame_diff > config.DISTANCE_CALC_MAX_FRAME_GAP)).cumsum()

            total_distance_m = 0.0
            active_time_sec = 0.0
            valid_speeds: List[float] = []
            walking_m = 0.0
            jogging_m = 0.0
            running_m = 0.0
            hsr_m = 0.0
            sprint_m = 0.0
            sprint_count = 0

            for _, seg in group.groupby(segment_ids):
                seg = seg.reset_index(drop=True)
                if len(seg) < 2:
                    continue

                sx = seg["pitch_x"].astype(float).rolling(
                    window=smooth_win, center=True, min_periods=1
                ).mean()
                sy = seg["pitch_y"].astype(float).rolling(
                    window=smooth_win, center=True, min_periods=1
                ).mean()

                seg_fdiff = seg["frame_idx"].diff()
                dx = sx.diff()
                dy = sy.diff()
                dist_step = (dx ** 2 + dy ** 2) ** 0.5

                if self.fps:
                    dt_step = seg_fdiff / float(self.fps)
                    speed_kmh = (dist_step / dt_step) * 3.6
                    valid_step = (
                        seg_fdiff.notna()
                        & (seg_fdiff <= config.DISTANCE_CALC_MAX_FRAME_GAP)
                        & (dt_step > 0)
                        & speed_kmh.notna()
                        & (speed_kmh <= config.MAX_REALISTIC_PLAYER_SPEED_KMH)
                    )

                    valid_dists = dist_step[valid_step]
                    valid_dts = dt_step[valid_step]
                    valid_spds = speed_kmh[valid_step]

                    total_distance_m += float(valid_dists.sum())
                    active_time_sec += float(valid_dts.sum())
                    valid_speeds.extend(valid_spds.tolist())

                    walking_m += float(valid_dists[valid_spds < config.SPEED_ZONE_WALKING_MAX_KMH].sum())
                    jogging_m += float(valid_dists[
                        (valid_spds >= config.SPEED_ZONE_WALKING_MAX_KMH)
                        & (valid_spds < config.SPEED_ZONE_JOGGING_MAX_KMH)
                    ].sum())
                    running_m += float(valid_dists[
                        (valid_spds >= config.SPEED_ZONE_JOGGING_MAX_KMH)
                        & (valid_spds < config.SPEED_ZONE_RUNNING_MAX_KMH)
                    ].sum())
                    hsr_m += float(valid_dists[
                        (valid_spds >= config.SPEED_ZONE_RUNNING_MAX_KMH)
                        & (valid_spds < config.SPEED_ZONE_HSR_MAX_KMH)
                    ].sum())
                    sprint_m += float(valid_dists[valid_spds >= config.SPEED_ZONE_HSR_MAX_KMH].sum())

                    # Ardışık sprint adımlarını say
                    is_sprint = (valid_step & (speed_kmh >= config.SPEED_ZONE_HSR_MAX_KMH)).tolist()
                    run_len = 0
                    for flag in is_sprint:
                        if flag:
                            run_len += 1
                        else:
                            if run_len >= min_sprint_frames:
                                sprint_count += 1
                            run_len = 0
                    if run_len >= min_sprint_frames:
                        sprint_count += 1
                else:
                    valid_step = (
                        seg_fdiff.notna()
                        & (seg_fdiff <= config.DISTANCE_CALC_MAX_FRAME_GAP)
                        & dist_step.notna()
                    )
                    total_distance_m += float(dist_step[valid_step].sum())

            if self.fps and valid_speeds:
                max_speed_kmh = round(max(valid_speeds), 2)
                avg_speed_kmh = (
                    round((total_distance_m / active_time_sec) * 3.6, 2)
                    if active_time_sec > 0
                    else 0.0
                )
            else:
                max_speed_kmh = None
                avg_speed_kmh = None

            team = group["team"].dropna().iloc[0] if group["team"].notna().any() else None
            class_name = group["class_name"].dropna().iloc[0] if group["class_name"].notna().any() else None

            summary_rows.append({
                "track_id": track_id,
                "class_name": class_name,
                "team": team,
                "frames_tracked": len(group),
                "active_time_sec": round(active_time_sec, 2) if self.fps else None,
                "total_distance_m": round(total_distance_m, 1),
                "avg_speed_kmh": avg_speed_kmh,
                "max_speed_kmh": max_speed_kmh,
                "walking_m": round(walking_m, 1) if self.fps else None,
                "jogging_m": round(jogging_m, 1) if self.fps else None,
                "running_m": round(running_m, 1) if self.fps else None,
                "high_speed_running_m": round(hsr_m, 1) if self.fps else None,
                "sprint_m": round(sprint_m, 1) if self.fps else None,
                "sprint_count": sprint_count if self.fps else None,
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("total_distance_m", ascending=False)

        if output_path is None:
            os.makedirs(config.EXPORT_DIR, exist_ok=True)
            output_path = os.path.join(config.EXPORT_DIR, config.EXPORT_DISTANCE_SUMMARY_FILENAME)

        summary_df.to_csv(output_path, index=False)
        print(f"[DataExporter] {len(summary_df)} oyuncu için mesafe özeti yazıldı: {output_path}")
        return output_path
