"""
data_exporter.py
------------------
ADIM 6: VERİ DIŞA AKTARIM (Pandas / CSV / SQL)

AMAÇ: Pipeline'ın önceki adımlarında (2-5) her karede ürettiğimiz zengin bilgiyi
(kim nerede, hangi takımda, saha üzerinde tam olarak hangi metrede) kalıcı,
analiz edilebilir bir forma dönüştürmek. Video bittiğinde elimizde sadece
"annotated video" değil, Excel'de açılabilen bir CSV ya da SQL sorgusu
yazılabilen bir veritabanı da olmalı.

TASARIM:
    - `flush_interval == 0` (varsayılan / geriye dönük uyumlu mod): Tüm kayıtlar
      bellekte `self.records` içinde tutulur, video bitince `export_to_csv()` /
      `export_to_sql()` ile diske yazılır.
    - `flush_interval > 0` (akış / streaming modu, örn. 2500 kare): 90 dakikalık
      maçlarda (~135.000 kare / ~2,7 milyon satır) bellek taşmasını önlemek için
      tamamlanan kare dilimleri periyodik olarak SQLite (WAL modu + toplu
      `executemany`) ve/veya eklemeli (append) CSV'ye aktarılıp `self.records`
      boşaltılır; `BALL_INTERPOLATE_MAX_GAP` kadar kayan kuyruk tamponu ve
      parçalar arası takım geriye-doldurma (backfill) korunur.
"""

import csv
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.perspective_transformer import PerspectiveTransformer
from modules.security import sanitize_csv_value, validate_sql_identifier


def _is_null(v) -> bool:
    """Hızlı ve NumPy-güvenli boşluk (None / NaN) kontrolü."""
    if v is None:
        return True
    if isinstance(v, (int, str, bool)):
        return False
    if isinstance(v, float):
        return math.isnan(v)
    if isinstance(v, np.floating):
        return bool(np.isnan(v))
    if isinstance(v, (np.integer, np.bool_)):
        return False
    return bool(pd.isna(v))


def _to_py_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if _is_null(v):
        return None
    f = float(v)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _to_py_str(v) -> Optional[str]:
    if _is_null(v):
        return None
    return str(v)


def _to_sqlite_bool_int(v) -> Optional[int]:
    if _is_null(v):
        return None
    return 1 if bool(v) else 0


def _to_csv_cell(v, is_bool: bool = False):
    if _is_null(v):
        return ""
    if is_bool:
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return "" if (math.isnan(f) or math.isinf(f)) else f
    return v


def _human_class_names() -> Set[str]:
    """4 sınıflı futbol taksonomisi ile eski COCO 'person' verilerini birlikte kabul eder."""
    return set(config.HUMAN_CLASS_NAMES) | {"person"}


class DataExporter:
    """
    Kare-kare takip verisini biriktirip (veya akış halinde diske boşaltıp)
    CSV/SQL'e yazan ve saha koordinatları üzerinden oyuncu mesafe/hız özeti
    çıkaran sınıf.
    """

    COLUMNS: Tuple[str, ...] = (
        "frame_idx",
        "timestamp_sec",
        "track_id",
        "class_name",
        "team",
        "confidence",
        "is_predicted",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "pixel_x",
        "pixel_y",
        "pitch_x",
        "pitch_y",
        "on_pitch",
    )

    SUMMARY_COLUMNS: Tuple[str, ...] = (
        "track_id",
        "class_name",
        "team",
        "frames_tracked",
        "active_time_sec",
        "total_distance_m",
        "avg_speed_kmh",
        "max_speed_kmh",
        "walking_m",
        "jogging_m",
        "running_m",
        "high_speed_running_m",
        "sprint_m",
        "sprint_count",
    )

    _SQL_BATCH_SIZE = 50_000

    def __init__(
        self,
        fps: Optional[float] = None,
        output_dir: Optional[str] = None,
        flush_interval: Optional[int] = None,
        csv_path: Optional[str] = None,
        sql_path: Optional[str] = None,
        table_name: Optional[str] = None,
        enable_csv: bool = True,
        enable_sql: bool = True,
        max_ball_gap: Optional[int] = None,
    ):
        """
        Parametreler
        ------------
        fps : float, optional
            Videonun saniyedeki kare sayısı (VideoProcessor.fps).
        output_dir : str, optional
            Verilirse varsayılan CSV/SQL/özet dosya yolları bu dizin altında çözülür.
        flush_interval : int, optional
            > 0 ise her bu kadar karede bir tamamlanan veriler diske akıtılır.
            None ise `config.EXPORT_FLUSH_INTERVAL_FRAMES` (varsayılan 0) okunur.
        csv_path / sql_path : str, optional
            Akış modunda kullanılacak açık CSV / SQLite dosya yolları.
        table_name : str, optional
            SQLite tablo adı (varsayılan: `config.EXPORT_SQL_TABLE_NAME`).
        enable_csv / enable_sql : bool
            Akış modunda CSV ve/veya SQLite çıktı hedeflerini etkinleştirir.
        max_ball_gap : int, optional
            Top interpolasyonu için maksimum kare boşluğu (kayan kuyruk tamponu).
        """
        self.fps = fps
        self.records: List[Dict] = []

        resolved_interval = (
            getattr(config, "EXPORT_FLUSH_INTERVAL_FRAMES", 0)
            if flush_interval is None
            else flush_interval
        )
        self.flush_interval = max(0, int(resolved_interval or 0))
        self.enable_csv = bool(enable_csv)
        self.enable_sql = bool(enable_sql)
        if self.flush_interval > 0 and not self.enable_csv and not self.enable_sql:
            self.flush_interval = 0

        self._output_dir_arg: Optional[str] = os.fspath(output_dir) if output_dir else None
        self._csv_path_arg: Optional[str] = os.fspath(csv_path) if csv_path else None
        self._sql_path_arg: Optional[str] = os.fspath(sql_path) if sql_path else None
        self._table_name_arg: Optional[str] = table_name
        self.max_ball_gap = int(
            config.BALL_INTERPOLATE_MAX_GAP if max_ball_gap is None else max_ball_gap
        )

        # Akış (streaming) durum değişkenleri
        self._frames_since_flush: int = 0
        self._last_appended_frame_idx: Optional[int] = None
        self._last_flushed_cutoff: Optional[int] = None
        self._has_flushed_any: bool = False
        self._stream_finalized: bool = False
        self._flushed_row_count: int = 0

        self._sql_conn: Optional[sqlite3.Connection] = None
        self._sql_table_created: bool = False
        self._active_sql_path: Optional[str] = None
        self._active_table_name: Optional[str] = None

        self._csv_header_written: bool = False
        self._active_csv_path: Optional[str] = None

        self._team_counts_by_tid: Dict[str, Counter] = {}
        self._flushed_null_team_tids: Set[str] = set()
        self._pre_flushed_backfill_count: int = 0
        self._pre_flushed_ball_interp_count: int = 0
        self._last_completed_real_ball_frame: Optional[int] = None

    # ----------------------------------------------------------------
    def _base_export_dir(self) -> str:
        return self._output_dir_arg if self._output_dir_arg is not None else config.EXPORT_DIR

    def _get_csv_path(self, override_path: Optional[str] = None) -> str:
        if override_path is not None:
            return os.fspath(override_path)
        if self._csv_path_arg is not None:
            return self._csv_path_arg
        return os.path.join(self._base_export_dir(), config.EXPORT_CSV_FILENAME)

    def _get_sql_path(self, override_path: Optional[str] = None) -> str:
        if override_path is not None:
            return os.fspath(override_path)
        if self._sql_path_arg is not None:
            return self._sql_path_arg
        return os.path.join(self._base_export_dir(), config.EXPORT_SQL_FILENAME)

    def _get_table_name(self, override_name: Optional[str] = None) -> str:
        if override_name is not None:
            raw_name = str(override_name)
        elif self._table_name_arg is not None:
            raw_name = str(self._table_name_arg)
        else:
            raw_name = config.EXPORT_SQL_TABLE_NAME
        return validate_sql_identifier(raw_name)

    def _get_summary_path(self, override_path: Optional[str] = None) -> str:
        if override_path is not None:
            return os.fspath(override_path)
        return os.path.join(self._base_export_dir(), config.EXPORT_DISTANCE_SUMMARY_FILENAME)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ----------------------------------------------------------------
    def close(self) -> None:
        """Açık SQLite bağlantısını WAL checkpoint yaparak güvenli biçimde kapatır."""
        if self._sql_conn is not None:
            try:
                self._sql_conn.commit()
                self._sql_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.Error:
                pass
            finally:
                self._sql_conn.close()
                self._sql_conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ----------------------------------------------------------------
    @classmethod
    def _to_sqlite_row(cls, rec: Dict) -> Tuple:
        return (
            int(rec["frame_idx"]),
            _to_py_float(rec.get("timestamp_sec")),
            _to_py_str(rec.get("track_id")),
            _to_py_str(rec.get("class_name")),
            _to_py_str(rec.get("team")),
            _to_py_float(rec.get("confidence")),
            _to_sqlite_bool_int(rec.get("is_predicted")),
            _to_py_float(rec.get("bbox_x1")),
            _to_py_float(rec.get("bbox_y1")),
            _to_py_float(rec.get("bbox_x2")),
            _to_py_float(rec.get("bbox_y2")),
            _to_py_float(rec.get("pixel_x")),
            _to_py_float(rec.get("pixel_y")),
            _to_py_float(rec.get("pitch_x")),
            _to_py_float(rec.get("pitch_y")),
            _to_sqlite_bool_int(rec.get("on_pitch")),
        )

    @classmethod
    def _to_csv_row(cls, rec: Dict) -> List:
        return [
            int(rec["frame_idx"]),
            _to_csv_cell(rec.get("timestamp_sec")),
            sanitize_csv_value(_to_csv_cell(rec.get("track_id"))),
            sanitize_csv_value(_to_csv_cell(rec.get("class_name"))),
            sanitize_csv_value(_to_csv_cell(rec.get("team"))),
            _to_csv_cell(rec.get("confidence")),
            _to_csv_cell(rec.get("is_predicted"), is_bool=True),
            _to_csv_cell(rec.get("bbox_x1")),
            _to_csv_cell(rec.get("bbox_y1")),
            _to_csv_cell(rec.get("bbox_x2")),
            _to_csv_cell(rec.get("bbox_y2")),
            _to_csv_cell(rec.get("pixel_x")),
            _to_csv_cell(rec.get("pixel_y")),
            _to_csv_cell(rec.get("pitch_x")),
            _to_csv_cell(rec.get("pitch_y")),
            _to_csv_cell(rec.get("on_pitch"), is_bool=True),
        ]

    @staticmethod
    def _create_sql_table(conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(
            f'CREATE TABLE "{table_name}" ('
            "frame_idx INTEGER, "
            "timestamp_sec REAL, "
            "track_id TEXT, "
            "class_name TEXT, "
            "team TEXT, "
            "confidence REAL, "
            "is_predicted INTEGER, "
            "bbox_x1 REAL, "
            "bbox_y1 REAL, "
            "bbox_x2 REAL, "
            "bbox_y2 REAL, "
            "pixel_x REAL, "
            "pixel_y REAL, "
            "pitch_x REAL, "
            "pitch_y REAL, "
            "on_pitch INTEGER"
            ")"
        )

    @staticmethod
    def _insert_sql_sql_rows(
        conn: sqlite3.Connection,
        table_name: str,
        rows: List[Tuple],
        batch_size: int = 50_000,
    ) -> None:
        if not rows:
            return
        placeholders = ",".join(["?"] * len(DataExporter.COLUMNS))
        cols_sql = ",".join(DataExporter.COLUMNS)
        insert_sql = f'INSERT INTO "{table_name}" ({cols_sql}) VALUES ({placeholders})'
        with conn:
            for start in range(0, len(rows), batch_size):
                conn.executemany(insert_sql, rows[start : start + batch_size])

    def _ensure_sql_conn(self) -> sqlite3.Connection:
        if self._sql_conn is None:
            sql_path = self._get_sql_path()
            table_name = self._get_table_name()
            self._ensure_parent_dir(sql_path)
            conn = sqlite3.connect(sql_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA cache_size=-65536;")
            self._create_sql_table(conn, table_name)
            conn.commit()
            self._sql_conn = conn
            self._sql_table_created = True
            self._active_sql_path = sql_path
            self._active_table_name = table_name
        return self._sql_conn

    def _create_sql_indexes(self, conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_track_id ON "{table_name}"(track_id)')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_frame_idx ON "{table_name}"(frame_idx)')

    # ----------------------------------------------------------------
    def append_frame_data(self, frame_idx: int, all_tracked: List[Dict]) -> None:
        """
        Bir karedeki tüm tespitleri (oyuncular + top) iç listeye ekler.
        `flush_interval > 0` ise periyodik olarak tamamlanan kareleri diske boşaltır.
        """
        f_idx = int(frame_idx)
        if (
            self.flush_interval > 0
            and self._last_flushed_cutoff is not None
            and f_idx <= self._last_flushed_cutoff
        ):
            raise ValueError(
                f"Akış modunda azalan frame_idx={f_idx} geldi "
                f"(son boşaltılan sınır={self._last_flushed_cutoff})."
            )

        timestamp_sec = (f_idx / self.fps) if self.fps else None
        ts_rounded = round(float(timestamp_sec), 3) if timestamp_sec is not None else None

        for det in all_tracked:
            x1, y1, x2, y2 = det["bbox"]
            fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
            foot_x, foot_y = PerspectiveTransformer.get_foot_position([fx1, fy1, fx2, fy2])

            class_name = det.get("class_name")
            team = det.get("team")
            track_id = det.get("track_id")

            if self.flush_interval > 0 and class_name != config.BALL_CLASS_NAME:
                if not _is_null(team):
                    tid_key = str(track_id)
                    counter = self._team_counts_by_tid.get(tid_key)
                    if counter is None:
                        counter = Counter()
                        self._team_counts_by_tid[tid_key] = counter
                    counter[str(team)] += 1

            self.records.append({
                "frame_idx": f_idx,
                "timestamp_sec": ts_rounded,
                "track_id": track_id,
                "class_name": class_name,
                "team": team,
                "confidence": det.get("confidence"),
                "is_predicted": det.get("is_predicted"),
                "bbox_x1": round(fx1, 1),
                "bbox_y1": round(fy1, 1),
                "bbox_x2": round(fx2, 1),
                "bbox_y2": round(fy2, 1),
                "pixel_x": round(float(foot_x), 1),
                "pixel_y": round(float(foot_y), 1),
                "pitch_x": det.get("pitch_x"),
                "pitch_y": det.get("pitch_y"),
                "on_pitch": det.get("on_pitch"),
            })

        if self.flush_interval > 0:
            if f_idx != self._last_appended_frame_idx:
                self._frames_since_flush += 1
                self._last_appended_frame_idx = f_idx
            if self._frames_since_flush >= self.flush_interval:
                self._flush_chunk(final=False)

    # ----------------------------------------------------------------
    @staticmethod
    def _needs_team_backfill(rec: Dict, resolved_team: Optional[str] = None) -> bool:
        """Eksik (None/NaN) takım etiketleri ile kilitsiz kaleci yer tutucularını belirler."""
        team = rec.get("team")
        if _is_null(team):
            return resolved_team is None or not _is_null(resolved_team)
        if rec.get("class_name") == "goalkeeper" and team == "Goalkeeper":
            return resolved_team is None or resolved_team != "Goalkeeper"
        return False

    def _build_team_lookup(self, team_map: Optional[Dict] = None) -> Dict[str, str]:
        counts_by_tid: Dict[str, Counter] = {
            tid: Counter(counter) for tid, counter in self._team_counts_by_tid.items()
        }
        for rec in self.records:
            if rec.get("class_name") == config.BALL_CLASS_NAME:
                continue
            team = rec.get("team")
            if not _is_null(team):
                tid_key = str(rec.get("track_id"))
                counter = counts_by_tid.get(tid_key)
                if counter is None:
                    counter = Counter()
                    counts_by_tid[tid_key] = counter
                counter[str(team)] += 1

        lookup: Dict[str, str] = {}
        for tid, counter in counts_by_tid.items():
            if not counter:
                continue
            real_counts = Counter({k: v for k, v in counter.items() if k != "Goalkeeper"})
            if real_counts:
                lookup[tid] = real_counts.most_common(1)[0][0]
            else:
                lookup[tid] = counter.most_common(1)[0][0]

        if team_map:
            for tid, team in team_map.items():
                if not _is_null(team):
                    lookup[str(tid)] = str(team)

        return lookup

    # ----------------------------------------------------------------
    def backfill_teams(self, team_map: Optional[Dict] = None) -> int:
        """
        K-Means fit edilmeden önceki ilk karelerde `team=None` veya geçici
        `team='Goalkeeper'` kalan oyuncu/kaleci satırlarını, sonradan kesinleşen
        `track_id -> team` eşlemesiyle geriye dönük doldurur.

        Dönüş: doldurulan satır sayısı.
        """
        lookup = self._build_team_lookup(team_map=team_map)

        filled = 0
        for rec in self.records:
            if rec.get("class_name") == config.BALL_CLASS_NAME:
                continue
            if self._needs_team_backfill(rec):
                resolved = lookup.get(str(rec.get("track_id")))
                if resolved is not None and self._needs_team_backfill(rec, resolved):
                    rec["team"] = resolved
                    filled += 1

        if self._has_flushed_any and self._flushed_null_team_tids and lookup:
            patch_sql = 0
            patch_csv = 0
            if self.enable_sql and self._active_sql_path and os.path.exists(self._active_sql_path):
                patch_sql = self._patch_sql_teams(lookup)
            if self.enable_csv and self._active_csv_path and os.path.exists(self._active_csv_path):
                patch_csv = self._patch_csv_teams(lookup)
            filled += patch_sql if (self.enable_sql and self._active_sql_path) else patch_csv
            for tid in list(self._flushed_null_team_tids):
                if tid in lookup and lookup[tid] != "Goalkeeper":
                    self._flushed_null_team_tids.discard(tid)

        return filled

    # ----------------------------------------------------------------
    def _interpolate_ball_in_buffer(self, max_gap: int) -> int:
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
            if (
                self._last_completed_real_ball_frame is not None
                and f2 <= self._last_completed_real_ball_frame
            ):
                continue

            gap = f2 - f1
            if not (1 < gap <= max_gap):
                continue

            r1 = real_by_frame[f1]
            r2 = real_by_frame[f2]

            has_pitch = (
                not _is_null(r1.get("pitch_x"))
                and not _is_null(r1.get("pitch_y"))
                and not _is_null(r2.get("pitch_x"))
                and not _is_null(r2.get("pitch_y"))
            )

            for f in range(f1 + 1, f2):
                alpha = (f - f1) / gap
                bx1 = round(float(r1["bbox_x1"]) + alpha * (float(r2["bbox_x1"]) - float(r1["bbox_x1"])), 1)
                by1 = round(float(r1["bbox_y1"]) + alpha * (float(r2["bbox_y1"]) - float(r1["bbox_y1"])), 1)
                bx2 = round(float(r1["bbox_x2"]) + alpha * (float(r2["bbox_x2"]) - float(r1["bbox_x2"])), 1)
                by2 = round(float(r1["bbox_y2"]) + alpha * (float(r2["bbox_y2"]) - float(r1["bbox_y2"])), 1)
                foot_x, foot_y = PerspectiveTransformer.get_foot_position([bx1, by1, bx2, by2])
                pixel_x = round(float(foot_x), 1)
                pixel_y = round(float(foot_y), 1)

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

        if real_frames:
            if (
                self._last_completed_real_ball_frame is None
                or real_frames[-1] > self._last_completed_real_ball_frame
            ):
                self._last_completed_real_ball_frame = real_frames[-1]

        self.records.sort(key=lambda r: (int(r["frame_idx"]), str(r.get("track_id"))))
        return interpolated_count

    def interpolate_ball_positions(self, max_gap: int = config.BALL_INTERPOLATE_MAX_GAP) -> int:
        """
        Gerçek top tespitleri (is_predicted=False) arasındaki kısa boşlukları
        doğrusal olarak interpole eder; mevcut tahmin kayıtlarını günceller veya
        eksik ara kareler için yeni tahmin kayıtları ekler.

        Dönüş: interpole edilen kare sayısı.
        """
        return self._interpolate_ball_in_buffer(max_gap=max_gap)

    # ----------------------------------------------------------------
    def _write_chunk_to_targets(self, chunk: List[Dict]) -> None:
        if not chunk:
            return

        if self.enable_sql:
            conn = self._ensure_sql_conn()
            table_name = self._active_table_name or self._get_table_name()
            sql_rows = [self._to_sqlite_row(rec) for rec in chunk]
            self._insert_sql_sql_rows(conn, table_name, sql_rows, batch_size=self._SQL_BATCH_SIZE)

        if self.enable_csv:
            csv_path = self._active_csv_path or self._get_csv_path()
            self._ensure_parent_dir(csv_path)
            mode = "a" if self._csv_header_written else "w"
            with open(csv_path, mode, encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                if not self._csv_header_written:
                    writer.writerow(self.COLUMNS)
                    self._csv_header_written = True
                for rec in chunk:
                    writer.writerow(self._to_csv_row(rec))
            self._active_csv_path = csv_path

    def _flush_chunk(
        self,
        final: bool = False,
        skip_interp: bool = False,
        skip_backfill: bool = False,
    ) -> int:
        if not self.records:
            return 0

        if not final:
            # Takımlar henüz hiç öğrenilmediyse ilk flush'ı sınırlı bir üst limite kadar ertele
            if (
                not self._team_counts_by_tid
                and not self._has_flushed_any
                and self._frames_since_flush < self.flush_interval * 2
            ):
                return 0

            max_frame = max(int(r["frame_idx"]) for r in self.records)
            cutoff = max_frame - self.max_ball_gap - 1 if self.max_ball_gap >= 2 else max_frame
            if not any(int(r["frame_idx"]) <= cutoff for r in self.records):
                return 0
        else:
            cutoff = None

        if not skip_interp:
            self._pre_flushed_ball_interp_count += self._interpolate_ball_in_buffer(self.max_ball_gap)

        if cutoff is None:
            to_flush = list(self.records)
            to_keep: List[Dict] = []
        else:
            to_flush = []
            to_keep = []
            for r in self.records:
                if int(r["frame_idx"]) <= cutoff:
                    to_flush.append(r)
                else:
                    to_keep.append(r)

        if not to_flush:
            return 0

        to_flush.sort(key=lambda r: (int(r["frame_idx"]), str(r.get("track_id"))))

        if config.TEAM_CLASSIFIER_BACKFILL_TEAMS:
            if not skip_backfill:
                lookup = self._build_team_lookup()
                for rec in to_flush:
                    if rec.get("class_name") == config.BALL_CLASS_NAME:
                        continue
                    if self._needs_team_backfill(rec):
                        tid_key = str(rec.get("track_id"))
                        resolved = lookup.get(tid_key)
                        if resolved is not None and self._needs_team_backfill(rec, resolved):
                            rec["team"] = resolved
                            self._pre_flushed_backfill_count += 1
                        else:
                            self._flushed_null_team_tids.add(tid_key)
            else:
                for rec in to_flush:
                    if rec.get("class_name") != config.BALL_CLASS_NAME and self._needs_team_backfill(rec):
                        self._flushed_null_team_tids.add(str(rec.get("track_id")))

        self._write_chunk_to_targets(to_flush)

        self.records = to_keep
        self._has_flushed_any = True
        self._stream_finalized = False
        self._flushed_row_count += len(to_flush)
        if cutoff is not None:
            self._last_flushed_cutoff = cutoff
        elif to_flush:
            self._last_flushed_cutoff = max(int(r["frame_idx"]) for r in to_flush)
        self._frames_since_flush = len({int(r["frame_idx"]) for r in to_keep})
        return len(to_flush)

    def _patch_sql_teams(self, lookup: Dict[str, str]) -> int:
        sql_path = self._active_sql_path or self._get_sql_path()
        table_name = self._active_table_name or self._get_table_name()
        if not os.path.exists(sql_path):
            return 0

        total_updated = 0
        own_conn = self._sql_conn is None
        conn = sqlite3.connect(sql_path) if own_conn else self._sql_conn
        try:
            with conn:
                for tid in self._flushed_null_team_tids:
                    team = lookup.get(tid)
                    if team is None:
                        continue
                    if team != "Goalkeeper":
                        cur = conn.execute(
                            f'UPDATE "{table_name}" SET team = ? '
                            "WHERE track_id = ? AND class_name != ? "
                            "AND (team IS NULL OR (class_name = 'goalkeeper' AND team = 'Goalkeeper'))",
                            (team, tid, config.BALL_CLASS_NAME),
                        )
                    else:
                        cur = conn.execute(
                            f'UPDATE "{table_name}" SET team = ? '
                            "WHERE track_id = ? AND team IS NULL AND class_name != ?",
                            (team, tid, config.BALL_CLASS_NAME),
                        )
                    if cur.rowcount and cur.rowcount > 0:
                        total_updated += int(cur.rowcount)
        finally:
            if own_conn:
                conn.close()
        return total_updated

    def _patch_csv_teams(self, lookup: Dict[str, str]) -> int:
        csv_path = self._active_csv_path or self._get_csv_path()
        if not os.path.exists(csv_path):
            return 0
        target_tids = {tid for tid in self._flushed_null_team_tids if tid in lookup}
        if not target_tids:
            return 0

        tmp_csv = f"{csv_path}.tmp_patch"
        updated = 0
        with (
            open(csv_path, "r", encoding="utf-8", newline="") as fin,
            open(tmp_csv, "w", encoding="utf-8", newline="") as fout,
        ):
            reader = csv.reader(fin)
            writer = csv.writer(fout, lineterminator="\n")
            header = next(reader, None)
            if header is None:
                return 0
            writer.writerow(header)
            tid_idx = header.index("track_id")
            cls_idx = header.index("class_name")
            team_idx = header.index("team")
            for row in reader:
                if row[cls_idx] != config.BALL_CLASS_NAME and row[tid_idx] in target_tids:
                    resolved = lookup[row[tid_idx]]
                    is_empty = row[team_idx] == ""
                    is_gk_placeholder = (
                        row[cls_idx] == "goalkeeper"
                        and row[team_idx] == "Goalkeeper"
                        and resolved != "Goalkeeper"
                    )
                    if is_empty or is_gk_placeholder:
                        row[team_idx] = sanitize_csv_value(resolved)
                        updated += 1
                writer.writerow(row)
        os.replace(tmp_csv, csv_path)
        return updated

    # ----------------------------------------------------------------
    def _finalize_stream(self, team_map: Optional[Dict] = None) -> Dict[str, int]:
        interpolated_tail = self._interpolate_ball_in_buffer(self.max_ball_gap)
        total_interpolated = self._pre_flushed_ball_interp_count + interpolated_tail

        do_backfill = bool(config.TEAM_CLASSIFIER_BACKFILL_TEAMS)
        lookup = self._build_team_lookup(team_map=team_map) if do_backfill else {}

        tail_backfilled = 0
        if do_backfill and self.records and lookup:
            for rec in self.records:
                if rec.get("class_name") == config.BALL_CLASS_NAME:
                    continue
                if self._needs_team_backfill(rec):
                    resolved = lookup.get(str(rec.get("track_id")))
                    if resolved is not None and self._needs_team_backfill(rec, resolved):
                        rec["team"] = resolved
                        tail_backfilled += 1

        if self.records:
            self._flush_chunk(final=True, skip_interp=True, skip_backfill=True)
        elif not self._has_flushed_any:
            # Hiç tespit olmayan (0 satır) akışlarda bile boş CSV başlığını ve boş SQLite tablosunu hazırla
            if self.enable_csv and not self._csv_header_written:
                csv_path = self._active_csv_path or self._get_csv_path()
                self._ensure_parent_dir(csv_path)
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(self.COLUMNS)
                self._csv_header_written = True
                self._active_csv_path = csv_path
            if self.enable_sql and not self._sql_table_created:
                self._ensure_sql_conn()

        post_patch_sql = 0
        post_patch_csv = 0

        if self.enable_sql and self._sql_table_created:
            conn = self._ensure_sql_conn()
            table_name = self._active_table_name or self._get_table_name()
            self._create_sql_indexes(conn, table_name)
            conn.commit()
            if do_backfill and self._flushed_null_team_tids and lookup:
                post_patch_sql = self._patch_sql_teams(lookup)
            self.close()

        if (
            self.enable_csv
            and self._csv_header_written
            and do_backfill
            and self._flushed_null_team_tids
            and lookup
        ):
            post_patch_csv = self._patch_csv_teams(lookup)

        post_patch_count = (
            post_patch_sql if (self.enable_sql and self._sql_table_created) else post_patch_csv
        )
        total_backfilled = self._pre_flushed_backfill_count + tail_backfilled + post_patch_count

        self._flushed_null_team_tids.clear()
        self._stream_finalized = True

        return {
            "backfilled_teams": total_backfilled,
            "interpolated_ball_frames": total_interpolated,
        }

    # ----------------------------------------------------------------
    def postprocess(self, team_map: Optional[Dict] = None) -> Dict[str, int]:
        """
        Video işleme bittikten sonra veriyi temizler: ilk karelerdeki eksik
        takım etiketlerini geriye dönük doldurur ve top boşluklarını interpole eder.
        Akış modunda kalan tamponu boşaltır, indeksleri kurar ve dosyaları kapatır.
        """
        if self.flush_interval > 0 or self._has_flushed_any:
            stats = self._finalize_stream(team_map=team_map)
        else:
            backfilled = (
                self.backfill_teams(team_map=team_map)
                if config.TEAM_CLASSIFIER_BACKFILL_TEAMS
                else 0
            )
            interpolated = self.interpolate_ball_positions()
            stats = {
                "backfilled_teams": backfilled,
                "interpolated_ball_frames": interpolated,
            }

        print(
            f"[DataExporter] Postprocess tamamlandı: {stats['backfilled_teams']} takım etiketi "
            f"dolduruldu, {stats['interpolated_ball_frames']} top karesi interpole edildi."
        )
        return stats

    # ----------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Biriken (veya akışla diske yazılmış) tüm kayıtları bir pandas DataFrame olarak döndürür."""
        if not self._has_flushed_any:
            return pd.DataFrame(self.records)

        frames: List[pd.DataFrame] = []
        if self.enable_sql and self._active_sql_path and os.path.exists(self._active_sql_path):
            table_name = self._active_table_name or self._get_table_name()
            conn = sqlite3.connect(self._active_sql_path)
            try:
                df_disk = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                for col in ("is_predicted", "on_pitch"):
                    if col in df_disk.columns:
                        df_disk[col] = df_disk[col].map(
                            lambda v: None if _is_null(v) else bool(int(v))
                        )
                frames.append(df_disk)
            finally:
                conn.close()
        elif self.enable_csv and self._active_csv_path and os.path.exists(self._active_csv_path):
            frames.append(pd.read_csv(self._active_csv_path))

        if self.records:
            frames.append(pd.DataFrame(self.records))

        if not frames:
            return pd.DataFrame(columns=list(self.COLUMNS))
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    # ----------------------------------------------------------------
    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Ham kare-kare takip verisini CSV olarak diske yazar.

        Dönüş: yazılan dosyanın tam yolu.
        """
        target_path = self._get_csv_path(output_path)
        self._ensure_parent_dir(target_path)

        if self.flush_interval > 0 or self._has_flushed_any:
            if not self._stream_finalized and (self.records or self._sql_conn is not None):
                self._finalize_stream()

            if self.enable_csv and self._active_csv_path and os.path.exists(self._active_csv_path):
                if os.path.abspath(target_path) != os.path.abspath(self._active_csv_path):
                    shutil.copyfile(self._active_csv_path, target_path)
                print(f"[DataExporter] Akış CSV hazır: {target_path}")
                return target_path

            if self.enable_sql and self._active_sql_path and os.path.exists(self._active_sql_path):
                table_name = self._active_table_name or self._get_table_name()
                conn = sqlite3.connect(self._active_sql_path)
                total_rows = 0
                try:
                    header_written = False
                    for chunk_df in pd.read_sql_query(
                        f'SELECT * FROM "{table_name}"',
                        conn,
                        chunksize=self._SQL_BATCH_SIZE,
                    ):
                        for col in ("is_predicted", "on_pitch"):
                            if col in chunk_df.columns:
                                chunk_df[col] = chunk_df[col].map(
                                    lambda v: "" if _is_null(v) else bool(int(v))
                                )
                        for col in ("track_id", "class_name", "team"):
                            if col in chunk_df.columns:
                                chunk_df[col] = chunk_df[col].map(sanitize_csv_value)
                        chunk_df.to_csv(
                            target_path,
                            mode="w" if not header_written else "a",
                            header=not header_written,
                            index=False,
                        )
                        header_written = True
                        total_rows += len(chunk_df)
                    if not header_written:
                        pd.DataFrame(columns=list(self.COLUMNS)).to_csv(target_path, index=False)
                finally:
                    conn.close()
                print(f"[DataExporter] {total_rows} satır CSV'ye yazıldı: {target_path}")
                return target_path

        df = self.to_dataframe().copy()
        for col in ("track_id", "class_name", "team"):
            if col in df.columns:
                df[col] = df[col].map(sanitize_csv_value)
        df.to_csv(target_path, index=False)
        print(f"[DataExporter] {len(df)} satır CSV'ye yazıldı: {target_path}")
        return target_path

    # ----------------------------------------------------------------
    def export_to_sql(self, db_path: Optional[str] = None, table_name: Optional[str] = None) -> str:
        """
        Ham kare-kare takip verisini bir SQLite veritabanına yazar.

        Dönüş: yazılan veritabanı dosyasının tam yolu.
        """
        target_db = self._get_sql_path(db_path)
        target_table = self._get_table_name(table_name)
        self._ensure_parent_dir(target_db)

        if self.flush_interval > 0 or self._has_flushed_any:
            if not self._stream_finalized and (self.records or self._sql_conn is not None):
                self._finalize_stream()

            if self.enable_sql and self._active_sql_path and os.path.exists(self._active_sql_path):
                same_db = os.path.abspath(target_db) == os.path.abspath(self._active_sql_path)
                same_table = target_table == (self._active_table_name or self._get_table_name())
                if same_db and same_table:
                    print(
                        f"[DataExporter] Akış SQLite hazır: {target_db} (tablo: {target_table})"
                    )
                    return target_db
                if not same_db and same_table:
                    shutil.copyfile(self._active_sql_path, target_db)
                    print(
                        f"[DataExporter] Akış SQLite kopyalandı: {target_db} (tablo: {target_table})"
                    )
                    return target_db

            if self.enable_csv and self._active_csv_path and os.path.exists(self._active_csv_path):
                conn = sqlite3.connect(target_db)
                total_rows = 0
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    self._create_sql_table(conn, target_table)
                    for chunk_df in pd.read_csv(self._active_csv_path, chunksize=self._SQL_BATCH_SIZE):
                        rows = [
                            self._to_sqlite_row(rec)
                            for rec in chunk_df.to_dict(orient="records")
                        ]
                        self._insert_sql_sql_rows(conn, target_table, rows, batch_size=self._SQL_BATCH_SIZE)
                        total_rows += len(rows)
                    self._create_sql_indexes(conn, target_table)
                    conn.commit()
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                finally:
                    conn.close()
                print(
                    f"[DataExporter] {total_rows} satır SQLite veritabanına yazıldı: "
                    f"{target_db} (tablo: {target_table})"
                )
                return target_db

        df = self.to_dataframe()

        conn = sqlite3.connect(target_db)
        try:
            if df.empty:
                self._create_sql_table(conn, target_table)
            else:
                rows = [self._to_sqlite_row(rec) for rec in self.records]
                self._create_sql_table(conn, target_table)
                self._insert_sql_sql_rows(conn, target_table, rows, batch_size=self._SQL_BATCH_SIZE)

            self._create_sql_indexes(conn, target_table)
            conn.commit()
        finally:
            conn.close()

        print(f"[DataExporter] {len(df)} satır SQLite veritabanına yazıldı: {target_db} (tablo: {target_table})")
        return target_db

    # ----------------------------------------------------------------
    @staticmethod
    def _summarize_track_rows(
        track_id: str,
        frame_idxs: Iterable[int],
        class_names: Iterable[Optional[str]],
        teams: Iterable[Optional[str]],
        pitch_xs: Iterable[float],
        pitch_ys: Iterable[float],
        fps: Optional[float],
        smooth_win: int,
        min_sprint_frames: int,
        max_frame_gap: int,
        max_realistic_speed_kmh: float,
    ) -> Optional[Dict]:
        f_arr = np.asarray(frame_idxs, dtype=np.int64)
        if f_arr.size == 0:
            return None

        px_arr = np.asarray(pitch_xs, dtype=np.float64)
        py_arr = np.asarray(pitch_ys, dtype=np.float64)
        cls_list = list(class_names)
        team_list = list(teams)

        # frame_idx sırasına getir ve aynı karedeki yinelenen satırları ilkini koruyarak temizle
        if f_arr.size > 1 and np.any(np.diff(f_arr) <= 0):
            order = np.argsort(f_arr, kind="mergesort")
            f_arr = f_arr[order]
            px_arr = px_arr[order]
            py_arr = py_arr[order]
            cls_list = [cls_list[i] for i in order]
            team_list = [team_list[i] for i in order]

            keep_mask = np.ones(f_arr.size, dtype=bool)
            keep_mask[1:] = f_arr[1:] != f_arr[:-1]
            if not np.all(keep_mask):
                f_arr = f_arr[keep_mask]
                px_arr = px_arr[keep_mask]
                py_arr = py_arr[keep_mask]
                cls_list = [c for c, k in zip(cls_list, keep_mask) if k]
                team_list = [t for t, k in zip(team_list, keep_mask) if k]

        first_team = next((t for t in team_list if not _is_null(t)), None)
        first_class = next((c for c in cls_list if not _is_null(c)), None)

        total_distance_m = 0.0
        active_time_sec = 0.0
        max_speed_val = 0.0
        has_valid_speed = False
        walking_m = 0.0
        jogging_m = 0.0
        running_m = 0.0
        hsr_m = 0.0
        sprint_m = 0.0
        sprint_count = 0

        n_frames = int(f_arr.size)
        if n_frames >= 2:
            frame_diffs = np.diff(f_arr)
            split_indices = np.flatnonzero(frame_diffs > max_frame_gap) + 1
            seg_starts = np.concatenate(([0], split_indices))
            seg_ends = np.concatenate((split_indices, [n_frames]))

            for s_start, s_end in zip(seg_starts.tolist(), seg_ends.tolist()):
                if s_end - s_start < 2:
                    continue

                seg_f = f_arr[s_start:s_end]
                seg_px = px_arr[s_start:s_end]
                seg_py = py_arr[s_start:s_end]

                if smooth_win > 1:
                    sx = (
                        pd.Series(seg_px, dtype=np.float64)
                        .rolling(window=smooth_win, center=True, min_periods=1)
                        .mean()
                        .to_numpy(dtype=np.float64)
                    )
                    sy = (
                        pd.Series(seg_py, dtype=np.float64)
                        .rolling(window=smooth_win, center=True, min_periods=1)
                        .mean()
                        .to_numpy(dtype=np.float64)
                    )
                else:
                    sx = seg_px
                    sy = seg_py

                seg_fdiff = np.diff(seg_f)
                dx = np.diff(sx)
                dy = np.diff(sy)
                dist_step = np.hypot(dx, dy)

                if fps:
                    dt_step = seg_fdiff / float(fps)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        speed_kmh = (dist_step / dt_step) * 3.6
                    valid_step = (
                        (seg_fdiff <= max_frame_gap)
                        & (dt_step > 0.0)
                        & np.isfinite(speed_kmh)
                        & (speed_kmh <= max_realistic_speed_kmh)
                    )

                    if np.any(valid_step):
                        v_dists = dist_step[valid_step]
                        v_dts = dt_step[valid_step]
                        v_spds = speed_kmh[valid_step]

                        total_distance_m += float(np.sum(v_dists))
                        active_time_sec += float(np.sum(v_dts))
                        seg_max_spd = float(np.max(v_spds))
                        if not has_valid_speed or seg_max_spd > max_speed_val:
                            max_speed_val = seg_max_spd
                        has_valid_speed = True

                        walking_m += float(np.sum(v_dists[v_spds < config.SPEED_ZONE_WALKING_MAX_KMH]))
                        jogging_m += float(
                            np.sum(
                                v_dists[
                                    (v_spds >= config.SPEED_ZONE_WALKING_MAX_KMH)
                                    & (v_spds < config.SPEED_ZONE_JOGGING_MAX_KMH)
                                ]
                            )
                        )
                        running_m += float(
                            np.sum(
                                v_dists[
                                    (v_spds >= config.SPEED_ZONE_JOGGING_MAX_KMH)
                                    & (v_spds < config.SPEED_ZONE_RUNNING_MAX_KMH)
                                ]
                            )
                        )
                        hsr_m += float(
                            np.sum(
                                v_dists[
                                    (v_spds >= config.SPEED_ZONE_RUNNING_MAX_KMH)
                                    & (v_spds < config.SPEED_ZONE_HSR_MAX_KMH)
                                ]
                            )
                        )
                        sprint_m += float(np.sum(v_dists[v_spds >= config.SPEED_ZONE_HSR_MAX_KMH]))

                    is_sprint = valid_step & (speed_kmh >= config.SPEED_ZONE_HSR_MAX_KMH)
                    if np.any(is_sprint):
                        padded = np.concatenate(([False], is_sprint, [False]))
                        changes = np.diff(padded.astype(np.int8))
                        run_starts = np.flatnonzero(changes == 1)
                        run_ends = np.flatnonzero(changes == -1)
                        sprint_count += int(np.sum((run_ends - run_starts) >= min_sprint_frames))
                else:
                    valid_step = (seg_fdiff <= max_frame_gap) & np.isfinite(dist_step)
                    if np.any(valid_step):
                        total_distance_m += float(np.sum(dist_step[valid_step]))

        if fps and has_valid_speed:
            max_speed_kmh = round(max_speed_val, 2)
            avg_speed_kmh = (
                round((total_distance_m / active_time_sec) * 3.6, 2)
                if active_time_sec > 0.0
                else 0.0
            )
        else:
            max_speed_kmh = None
            avg_speed_kmh = None

        return {
            "track_id": track_id,
            "class_name": first_class,
            "team": first_team,
            "frames_tracked": n_frames,
            "active_time_sec": round(active_time_sec, 2) if fps else None,
            "total_distance_m": round(total_distance_m, 1),
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "walking_m": round(walking_m, 1) if fps else None,
            "jogging_m": round(jogging_m, 1) if fps else None,
            "running_m": round(running_m, 1) if fps else None,
            "high_speed_running_m": round(hsr_m, 1) if fps else None,
            "sprint_m": round(sprint_m, 1) if fps else None,
            "sprint_count": sprint_count if fps else None,
        }

    # ----------------------------------------------------------------
    def _summarize_from_sqlite(
        self,
        sql_path: str,
        table_name: str,
        human_classes: Set[str],
        smooth_win: int,
        min_sprint_frames: int,
    ) -> List[Dict]:
        sorted_classes = sorted(human_classes)
        placeholders = ",".join(["?"] * len(sorted_classes))
        where_clauses = [
            f"class_name IN ({placeholders})",
            "pitch_x IS NOT NULL",
            "pitch_y IS NOT NULL",
            "(on_pitch IS NULL OR on_pitch != 0)",
        ]
        if config.DISTANCE_CALC_EXCLUDE_PREDICTED:
            where_clauses.append("(is_predicted IS NULL OR is_predicted != 1)")

        query = (
            f'SELECT track_id, frame_idx, class_name, team, pitch_x, pitch_y '
            f'FROM "{table_name}" '
            f'WHERE {" AND ".join(where_clauses)} '
            f'ORDER BY track_id'
        )

        summary_rows: List[Dict] = []
        conn = sqlite3.connect(sql_path)
        try:
            cur = conn.execute(query, tuple(sorted_classes))
            current_tid: Optional[str] = None
            f_buf: List[int] = []
            cls_buf: List[Optional[str]] = []
            team_buf: List[Optional[str]] = []
            px_buf: List[float] = []
            py_buf: List[float] = []

            def _flush_tid():
                if current_tid is None or not f_buf:
                    return
                row = self._summarize_track_rows(
                    track_id=current_tid,
                    frame_idxs=f_buf,
                    class_names=cls_buf,
                    teams=team_buf,
                    pitch_xs=px_buf,
                    pitch_ys=py_buf,
                    fps=self.fps,
                    smooth_win=smooth_win,
                    min_sprint_frames=min_sprint_frames,
                    max_frame_gap=config.DISTANCE_CALC_MAX_FRAME_GAP,
                    max_realistic_speed_kmh=config.MAX_REALISTIC_PLAYER_SPEED_KMH,
                )
                if row is not None:
                    summary_rows.append(row)

            while True:
                batch = cur.fetchmany(self._SQL_BATCH_SIZE)
                if not batch:
                    break
                for tid, fidx, cls_name, team, px, py in batch:
                    tid_str = str(tid)
                    if tid_str != current_tid:
                        _flush_tid()
                        current_tid = tid_str
                        f_buf.clear()
                        cls_buf.clear()
                        team_buf.clear()
                        px_buf.clear()
                        py_buf.clear()
                    f_buf.append(int(fidx))
                    cls_buf.append(cls_name)
                    team_buf.append(team)
                    px_buf.append(float(px))
                    py_buf.append(float(py))

            _flush_tid()
        finally:
            conn.close()

        return summary_rows

    def _summarize_from_csv_chunks(
        self,
        csv_path: str,
        human_classes: Set[str],
        smooth_win: int,
        min_sprint_frames: int,
    ) -> List[Dict]:
        usecols = [
            "track_id",
            "frame_idx",
            "class_name",
            "team",
            "is_predicted",
            "pitch_x",
            "pitch_y",
            "on_pitch",
        ]
        per_track: Dict[str, Dict[str, List]] = {}
        for chunk in pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype={"track_id": str, "class_name": str, "team": str},
            chunksize=200_000,
        ):
            mask = (
                chunk["class_name"].isin(human_classes)
                & chunk["pitch_x"].notna()
                & chunk["pitch_y"].notna()
                & (chunk["on_pitch"] != False)  # noqa: E712
            )
            if config.DISTANCE_CALC_EXCLUDE_PREDICTED:
                mask &= chunk["is_predicted"] != True  # noqa: E712

            filtered = chunk.loc[
                mask, ["track_id", "frame_idx", "class_name", "team", "pitch_x", "pitch_y"]
            ]
            if filtered.empty:
                continue

            for tid, fidx, cls_name, team, px, py in filtered.itertuples(index=False, name=None):
                tid_str = str(tid)
                bucket = per_track.get(tid_str)
                if bucket is None:
                    bucket = {"f": [], "c": [], "t": [], "px": [], "py": []}
                    per_track[tid_str] = bucket
                bucket["f"].append(int(fidx))
                bucket["c"].append(None if _is_null(cls_name) else cls_name)
                bucket["t"].append(None if _is_null(team) else team)
                bucket["px"].append(float(px))
                bucket["py"].append(float(py))

        summary_rows: List[Dict] = []
        for tid_str, bucket in per_track.items():
            row = self._summarize_track_rows(
                track_id=tid_str,
                frame_idxs=bucket["f"],
                class_names=bucket["c"],
                teams=bucket["t"],
                pitch_xs=bucket["px"],
                pitch_ys=bucket["py"],
                fps=self.fps,
                smooth_win=smooth_win,
                min_sprint_frames=min_sprint_frames,
                max_frame_gap=config.DISTANCE_CALC_MAX_FRAME_GAP,
                max_realistic_speed_kmh=config.MAX_REALISTIC_PLAYER_SPEED_KMH,
            )
            if row is not None:
                summary_rows.append(row)

        return summary_rows

    # ----------------------------------------------------------------
    def export_distance_summary(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Saha (metre) koordinatlarını kullanarak HER OYUNCU için toplam mesafe,
        aktif süre, ortalama/maksimum hız, FIFA hız bantları ve sprint sayısını
        hesaplayıp özet CSV'sine yazar.

        Dönüş: yazılan dosyanın tam yolu, ya da hesaplanacak veri yoksa None.
        """
        human_classes = _human_class_names()
        smooth_win = max(1, int(config.SPEED_SMOOTHING_WINDOW_FRAMES))
        min_sprint_frames = (
            max(2, int(round(self.fps * config.SPRINT_MIN_DURATION_SEC)))
            if self.fps
            else 2
        )

        summary_rows: List[Dict] = []

        if self._has_flushed_any or self.flush_interval > 0:
            if not self._stream_finalized and (self.records or self._sql_conn is not None):
                self._finalize_stream()

            if self.enable_sql and self._active_sql_path and os.path.exists(self._active_sql_path):
                table_name = self._active_table_name or self._get_table_name()
                summary_rows = self._summarize_from_sqlite(
                    sql_path=self._active_sql_path,
                    table_name=table_name,
                    human_classes=human_classes,
                    smooth_win=smooth_win,
                    min_sprint_frames=min_sprint_frames,
                )
            elif self.enable_csv and self._active_csv_path and os.path.exists(self._active_csv_path):
                summary_rows = self._summarize_from_csv_chunks(
                    csv_path=self._active_csv_path,
                    human_classes=human_classes,
                    smooth_win=smooth_win,
                    min_sprint_frames=min_sprint_frames,
                )
            elif self.records:
                # Akış modunda henüz hiç flush olmadıysa bellek içi yoldan hesapla
                pass
            else:
                print("[DataExporter] Özet çıkarılacak veri yok (records boş).")
                return None

        if not summary_rows and not self._has_flushed_any:
            if not self.records:
                print("[DataExporter] Özet çıkarılacak veri yok (records boş).")
                return None

            df = self.to_dataframe()
            if df.empty:
                print("[DataExporter] Özet çıkarılacak veri yok (records boş).")
                return None

            players_df = df[df["class_name"].isin(human_classes)].copy()
            players_df = players_df.dropna(subset=["pitch_x", "pitch_y"])
            if config.DISTANCE_CALC_EXCLUDE_PREDICTED:
                players_df = players_df[players_df["is_predicted"] != True]  # noqa: E712
            players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

            if players_df.empty:
                print("[DataExporter] Mesafe hesabı için yeterli saha-koordinatlı veri yok.")
                return None

            for track_id, group in players_df.groupby("track_id"):
                row = self._summarize_track_rows(
                    track_id=str(track_id),
                    frame_idxs=group["frame_idx"].to_numpy(dtype=np.int64),
                    class_names=group["class_name"].tolist(),
                    teams=group["team"].tolist(),
                    pitch_xs=group["pitch_x"].to_numpy(dtype=np.float64),
                    pitch_ys=group["pitch_y"].to_numpy(dtype=np.float64),
                    fps=self.fps,
                    smooth_win=smooth_win,
                    min_sprint_frames=min_sprint_frames,
                    max_frame_gap=config.DISTANCE_CALC_MAX_FRAME_GAP,
                    max_realistic_speed_kmh=config.MAX_REALISTIC_PLAYER_SPEED_KMH,
                )
                if row is not None:
                    summary_rows.append(row)

        if not summary_rows:
            print("[DataExporter] Mesafe hesabı için yeterli saha-koordinatlı veri yok.")
            return None

        summary_rows.sort(key=lambda r: (-float(r["total_distance_m"]), str(r["track_id"])))
        summary_df = pd.DataFrame(summary_rows, columns=list(self.SUMMARY_COLUMNS))
        for col in ("track_id", "class_name", "team"):
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].map(sanitize_csv_value)

        target_path = self._get_summary_path(output_path)
        self._ensure_parent_dir(target_path)

        summary_df.to_csv(target_path, index=False)
        print(f"[DataExporter] {len(summary_df)} oyuncu için mesafe özeti yazıldı: {target_path}")
        return target_path
