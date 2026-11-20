"""
match_analytics.py
------------------
ADIM 7: TAKTİK ANALİZ VE ISI HARİTASI MODÜLÜ

Kare-kare takip verisi (DataExporter çıktısı, tracking_data.csv veya .db)
üzerinden futbol odaklı taktiksel metrikler ve görseller üretir:
  - Topa sahip olma yüzdeleri (Team A vs Team B) ve oyuncu kontrol süreleri
  - Pas tespiti, pas isabeti, top kayıpları ve pas ağı (pass network)
  - Takım ortalama pozisyonları, saha yayılımı (width / depth) ve hat mesafeleri
  - OpenCV tabanlı 2D takım ve oyuncu ısı haritaları (heatmap PNG)

Hem kalibre edilmiş saha (metre) koordinatlarıyla (`pitch_x`, `pitch_y`),
hem de kalibrasyon yapılmamış kayıtlarda piksel koordinatlı (`pixel_x`, `pixel_y`)
fallback modunda sorunsuz çalışır.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def _clean_str(val: Any) -> Optional[str]:
    """NaN / None / boş değerleri güvenli biçimde None'a normalize eder."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _to_python_scalar(obj: Any) -> Any:
    """NumPy / Pandas tiplerini JSON serileştirilebilir standart Python tiplerine çevirir."""
    if isinstance(obj, dict):
        return {str(k): _to_python_scalar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_scalar(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bytes)) else False:
        return None
    return obj


class MatchAnalyzer:
    """
    Takip verisinden topa sahip olma, pas ağı, takım şekli ve ısı haritası
    üreten taktik analiz sınıfı.
    """

    def __init__(self, df: pd.DataFrame, fps: Optional[float] = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.fps = float(fps) if fps and fps > 0 else self._infer_fps()
        self.has_pitch_players, self.has_pitch_ball, self.is_calibrated = self._detect_calibration()

    # ----------------------------------------------------------------
    @classmethod
    def from_csv(cls, csv_path: str, fps: Optional[float] = None) -> "MatchAnalyzer":
        """CSV takip dosyasından MatchAnalyzer oluşturur."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        df = pd.read_csv(csv_path)
        return cls(df=df, fps=fps)

    # ----------------------------------------------------------------
    @classmethod
    def from_sql(
        cls,
        db_path: str,
        table_name: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> "MatchAnalyzer":
        """SQLite veritabanından MatchAnalyzer oluşturur."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite dosyası bulunamadı: {db_path}")
        table = table_name or config.EXPORT_SQL_TABLE_NAME
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        finally:
            conn.close()
        return cls(df=df, fps=fps)

    # ----------------------------------------------------------------
    def _infer_fps(self) -> float:
        """frame_idx ve timestamp_sec farklarından FPS değerini çıkarır (varsayılan: 25.0)."""
        if self.df.empty or "frame_idx" not in self.df.columns or "timestamp_sec" not in self.df.columns:
            return 25.0

        ts_df = (
            self.df[["frame_idx", "timestamp_sec"]]
            .dropna()
            .drop_duplicates(subset=["frame_idx"])
            .sort_values("frame_idx")
        )
        if len(ts_df) < 2:
            return 25.0

        df_idx = ts_df["frame_idx"].diff()
        dt = ts_df["timestamp_sec"].diff()
        valid = (df_idx > 0) & (dt > 0)
        if not valid.any():
            return 25.0

        est = float((df_idx[valid] / dt[valid]).median())
        if est <= 0 or math.isnan(est) or math.isinf(est):
            return 25.0
        if abs(est - round(est)) < 0.05:
            return float(round(est))
        return round(est, 2)

    # ----------------------------------------------------------------
    def _detect_calibration(self) -> Tuple[bool, bool, bool]:
        """Hem top hem oyuncu satırlarında geçerli saha (metre) koordinatı olup olmadığını kontrol eder."""
        if self.df.empty or "pitch_x" not in self.df.columns or "pitch_y" not in self.df.columns:
            return False, False, False

        has_valid_pitch = self.df["pitch_x"].notna() & self.df["pitch_y"].notna()
        is_ball = self.df["class_name"] == config.BALL_CLASS_NAME
        is_player = (
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (self.df["class_name"] != "referee")
            & (self.df["team"] != "Referee")
        )

        has_pitch_players = bool((has_valid_pitch & is_player).any())
        has_pitch_ball = bool((has_valid_pitch & is_ball).any())
        is_calibrated = bool(has_pitch_players and has_pitch_ball)
        return has_pitch_players, has_pitch_ball, is_calibrated

    # ----------------------------------------------------------------
    def compute_possession(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Her karede topa en yakın oyuncuyu belirler, çoğunluk oyuyla titreşimleri
        yumuşatır ve takım/oyuncu topa sahip olma istatistiklerini hesaplar.
        """
        coord_space = "pitch_meters" if self.is_calibrated else "pixel_fallback"
        x_col, y_col = ("pitch_x", "pitch_y") if self.is_calibrated else ("pixel_x", "pixel_y")
        max_dist = (
            float(config.POSSESSION_MAX_PLAYER_BALL_DIST_M)
            if self.is_calibrated
            else float(config.POSSESSION_MAX_PLAYER_BALL_DIST_PX)
        )

        empty_cols = [
            "frame_idx", "timestamp_sec", "ball_x", "ball_y",
            "raw_track_id", "raw_team", "nearest_dist",
            "smoothed_track_id", "smoothed_team",
        ]
        if self.df.empty or x_col not in self.df.columns or y_col not in self.df.columns:
            return pd.DataFrame(columns=empty_cols), {
                "mode": coord_space,
                "max_distance_threshold": max_dist,
                "total_frames": 0,
                "ball_detected_frames": 0,
                "controlled_frames": 0,
                "team_a_frames": 0,
                "team_b_frames": 0,
                "team_a_pct": 0.0,
                "team_b_pct": 0.0,
                "player_control": [],
            }

        total_frames = int(self.df["frame_idx"].nunique())

        ball_df = self.df[self.df["class_name"] == config.BALL_CLASS_NAME].dropna(subset=[x_col, y_col]).copy()
        if not ball_df.empty:
            # Aynı karede birden fazla kayıt varsa gerçek tespiti (is_predicted=False) tercih et
            if "is_predicted" in ball_df.columns:
                ball_df["_pred_sort"] = ball_df["is_predicted"].fillna(False).astype(bool)
                ball_df = ball_df.sort_values(["frame_idx", "_pred_sort"]).drop_duplicates(subset=["frame_idx"])
            else:
                ball_df = ball_df.sort_values("frame_idx").drop_duplicates(subset=["frame_idx"])

        players_df = self.df[
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (self.df["class_name"] != "referee")
            & (self.df["team"] != "Referee")
        ].dropna(subset=[x_col, y_col]).copy()
        if self.is_calibrated and "on_pitch" in players_df.columns:
            players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        if ball_df.empty:
            return pd.DataFrame(columns=empty_cols), {
                "mode": coord_space,
                "max_distance_threshold": max_dist,
                "total_frames": total_frames,
                "ball_detected_frames": 0,
                "controlled_frames": 0,
                "team_a_frames": 0,
                "team_b_frames": 0,
                "team_a_pct": 0.0,
                "team_b_pct": 0.0,
                "player_control": [],
            }

        players_by_frame = {int(fid): grp for fid, grp in players_df.groupby("frame_idx")}
        track_team_lookup: Dict[str, str] = {}
        for _, row in players_df.iterrows():
            t = _clean_str(row.get("team"))
            tid = _clean_str(row.get("track_id"))
            if tid and t:
                track_team_lookup.setdefault(tid, t)

        raw_rows: List[Dict[str, Any]] = []
        for _, brow in ball_df.iterrows():
            fid = int(brow["frame_idx"])
            bx = float(brow[x_col])
            by = float(brow[y_col])
            ts = brow.get("timestamp_sec")
            ts_val = round(float(ts), 3) if ts is not None and not pd.isna(ts) else round(fid / self.fps, 3)

            f_players = players_by_frame.get(fid)
            raw_tid = None
            raw_team = None
            nearest_dist = None

            if f_players is not None and not f_players.empty:
                pxs = f_players[x_col].to_numpy(dtype=float)
                pys = f_players[y_col].to_numpy(dtype=float)
                dists = np.hypot(pxs - bx, pys - by)
                min_idx = int(np.argmin(dists))
                min_d = float(dists[min_idx])
                nearest_dist = round(min_d, 3)
                if min_d <= max_dist:
                    chosen = f_players.iloc[min_idx]
                    raw_tid = _clean_str(chosen.get("track_id"))
                    raw_team = _clean_str(chosen.get("team")) or (
                        track_team_lookup.get(raw_tid) if raw_tid else None
                    )

            raw_rows.append({
                "frame_idx": fid,
                "timestamp_sec": ts_val,
                "ball_x": round(bx, 3),
                "ball_y": round(by, 3),
                "raw_track_id": raw_tid,
                "raw_team": raw_team,
                "nearest_dist": nearest_dist,
            })

        timeline_df = pd.DataFrame(raw_rows)

        # Çoğunluk oyuyla yumuşatma (majority vote smoothing)
        win = max(1, int(config.POSSESSION_SMOOTHING_FRAMES))
        half = win // 2
        n = len(timeline_df)
        raw_tids = timeline_df["raw_track_id"].tolist()
        raw_teams = timeline_df["raw_team"].tolist()

        smoothed_tids: List[Optional[str]] = []
        smoothed_teams: List[Optional[str]] = []

        for i in range(n):
            if win <= 1:
                smoothed_tids.append(raw_tids[i])
                smoothed_teams.append(raw_teams[i])
                continue

            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            slice_tids = [t for t in raw_tids[lo:hi] if t is not None]
            if not slice_tids:
                smoothed_tids.append(None)
                smoothed_teams.append(None)
                continue

            counts = Counter(slice_tids)
            top_tid, top_cnt = counts.most_common(1)[0]
            # Tek karelik boşluklarda köprü kur ama uzun boşlukları yapay uzatma
            min_required = 2 if raw_tids[i] is None and (hi - lo) >= 3 else 1
            if top_cnt >= min_required:
                smoothed_tids.append(top_tid)
                # Takım bilgisini önce track lookup'tan, yoksa ham satırlardan al
                s_team = track_team_lookup.get(top_tid)
                if s_team is None:
                    for k in range(lo, hi):
                        if raw_tids[k] == top_tid and raw_teams[k] is not None:
                            s_team = raw_teams[k]
                            break
                smoothed_teams.append(s_team)
            else:
                smoothed_tids.append(None)
                smoothed_teams.append(None)

        timeline_df["smoothed_track_id"] = smoothed_tids
        timeline_df["smoothed_team"] = smoothed_teams

        controlled_mask = timeline_df["smoothed_track_id"].notna()
        controlled_frames = int(controlled_mask.sum())
        team_a_frames = int((timeline_df["smoothed_team"] == "Team A").sum())
        team_b_frames = int((timeline_df["smoothed_team"] == "Team B").sum())
        ab_total = team_a_frames + team_b_frames

        team_a_pct = round((team_a_frames / ab_total) * 100.0, 2) if ab_total > 0 else 0.0
        team_b_pct = round((team_b_frames / ab_total) * 100.0, 2) if ab_total > 0 else 0.0

        player_control: List[Dict[str, Any]] = []
        ctrl_df = timeline_df[controlled_mask]
        if not ctrl_df.empty:
            for tid, grp in ctrl_df.groupby("smoothed_track_id"):
                t_val = _clean_str(grp["smoothed_team"].dropna().iloc[0]) if grp["smoothed_team"].notna().any() else None
                f_cnt = int(len(grp))
                player_control.append({
                    "track_id": tid,
                    "team": t_val,
                    "frames": f_cnt,
                    "seconds": round(f_cnt / self.fps, 2),
                })
            player_control.sort(key=lambda r: r["frames"], reverse=True)

        stats = {
            "mode": coord_space,
            "max_distance_threshold": max_dist,
            "total_frames": total_frames,
            "ball_detected_frames": int(len(timeline_df)),
            "controlled_frames": controlled_frames,
            "team_a_frames": team_a_frames,
            "team_b_frames": team_b_frames,
            "team_a_pct": team_a_pct,
            "team_b_pct": team_b_pct,
            "player_control": player_control,
        }
        return timeline_df, stats

    # ----------------------------------------------------------------
    def detect_passes(self, timeline_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Top kontrolündeki oyuncu geçişlerinden başarılı pasları, top kayıplarını
        ve takım pas ağını (pass network) çıkarır.
        """
        empty_summary = {
            "mode": "pitch_meters" if self.is_calibrated else "pixel_fallback",
            "total_completed_passes": 0,
            "total_turnovers": 0,
            "team_stats": {
                "Team A": {"completed_passes": 0, "turnovers": 0, "attempted_passes": 0, "pass_accuracy_pct": 0.0},
                "Team B": {"completed_passes": 0, "turnovers": 0, "attempted_passes": 0, "pass_accuracy_pct": 0.0},
            },
            "pass_network": [],
            "top_pass_connections": [],
            "events": [],
        }
        if timeline_df is None or timeline_df.empty:
            return empty_summary

        ctrl = timeline_df[timeline_df["smoothed_track_id"].notna()].reset_index(drop=True)
        if ctrl.empty:
            return empty_summary

        # Ardışık aynı track_id karelerini kontrol periyotlarına (spell) dönüştür
        spells: List[Dict[str, Any]] = []
        current_tid = None
        for _, row in ctrl.iterrows():
            tid = _clean_str(row["smoothed_track_id"])
            team = _clean_str(row["smoothed_team"])
            fid = int(row["frame_idx"])
            bx = float(row["ball_x"])
            by = float(row["ball_y"])

            if current_tid == tid and spells:
                spells[-1]["end_frame"] = fid
                spells[-1]["end_x"] = bx
                spells[-1]["end_y"] = by
                if spells[-1]["team"] is None and team is not None:
                    spells[-1]["team"] = team
            else:
                spells.append({
                    "track_id": tid,
                    "team": team,
                    "start_frame": fid,
                    "end_frame": fid,
                    "start_x": bx,
                    "start_y": by,
                    "end_x": bx,
                    "end_y": by,
                })
                current_tid = tid

        min_move = (
            float(config.PASS_MIN_BALL_MOVE_M)
            if self.is_calibrated
            else float(config.PASS_MIN_BALL_MOVE_PX)
        )

        events: List[Dict[str, Any]] = []
        edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        team_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"completed": 0, "turnovers": 0})

        for i in range(len(spells) - 1):
            s1 = spells[i]
            s2 = spells[i + 1]
            if s1["track_id"] == s2["track_id"]:
                continue

            gap_sec = (s2["start_frame"] - s1["end_frame"]) / self.fps
            move_dist = math.hypot(s2["start_x"] - s1["end_x"], s2["start_y"] - s1["end_y"])

            if gap_sec > config.PASS_MAX_GAP_SEC or move_dist < min_move:
                continue
            if s1["team"] is None or s2["team"] is None:
                continue

            if s1["team"] == s2["team"]:
                event_type = "completed_pass"
                team_counts[s1["team"]]["completed"] += 1
                edge_counts[(s1["track_id"], s2["track_id"], s1["team"])] += 1
            else:
                event_type = "turnover"
                team_counts[s1["team"]]["turnovers"] += 1

            events.append({
                "event_type": event_type,
                "from_track_id": s1["track_id"],
                "to_track_id": s2["track_id"],
                "from_team": s1["team"],
                "to_team": s2["team"],
                "start_frame": s1["end_frame"],
                "end_frame": s2["start_frame"],
                "duration_sec": round(gap_sec, 2),
                "distance": round(move_dist, 2),
            })

        team_stats: Dict[str, Dict[str, Any]] = {}
        for tname in ("Team A", "Team B"):
            comp = team_counts[tname]["completed"]
            turn = team_counts[tname]["turnovers"]
            att = comp + turn
            acc = round((comp / att) * 100.0, 2) if att > 0 else 0.0
            team_stats[tname] = {
                "completed_passes": comp,
                "turnovers": turn,
                "attempted_passes": att,
                "pass_accuracy_pct": acc,
            }

        pass_network = [
            {
                "from_track_id": f_tid,
                "to_track_id": t_tid,
                "team": t_name,
                "count": cnt,
            }
            for (f_tid, t_tid, t_name), cnt in edge_counts.items()
        ]
        pass_network.sort(key=lambda e: e["count"], reverse=True)

        return {
            "mode": "pitch_meters" if self.is_calibrated else "pixel_fallback",
            "total_completed_passes": sum(1 for e in events if e["event_type"] == "completed_pass"),
            "total_turnovers": sum(1 for e in events if e["event_type"] == "turnover"),
            "team_stats": team_stats,
            "pass_network": pass_network,
            "top_pass_connections": pass_network[:10],
            "events": events,
        }

    # ----------------------------------------------------------------
    def compute_team_shape(self) -> Dict[str, Any]:
        """
        Team A ve Team B için ortalama pozisyon (centroid), saha genişliği/derinliği
        ve oyuncu bazlı ortalama pozisyonları hesaplar.
        """
        use_pitch = self.has_pitch_players
        x_col, y_col = ("pitch_x", "pitch_y") if use_pitch else ("pixel_x", "pixel_y")
        mode = "pitch_meters" if use_pitch else "pixel_fallback"

        result: Dict[str, Any] = {"mode": mode, "teams": {}}
        if self.df.empty or x_col not in self.df.columns or y_col not in self.df.columns:
            return result

        players_df = self.df[
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (self.df["class_name"] != "referee")
            & (self.df["team"].isin(["Team A", "Team B"]))
        ].dropna(subset=[x_col, y_col]).copy()

        if use_pitch and "on_pitch" in players_df.columns:
            players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        for tname in ("Team A", "Team B"):
            t_df = players_df[players_df["team"] == tname]
            if t_df.empty:
                result["teams"][tname] = None
                continue

            player_positions: List[Dict[str, Any]] = []
            for tid, p_grp in t_df.groupby("track_id"):
                player_positions.append({
                    "track_id": str(tid),
                    "avg_x": round(float(p_grp[x_col].mean()), 2),
                    "avg_y": round(float(p_grp[y_col].mean()), 2),
                    "frames": int(len(p_grp)),
                })
            player_positions.sort(key=lambda p: p["frames"], reverse=True)

            xs = np.array([p["avg_x"] for p in player_positions], dtype=float)
            ys = np.array([p["avg_y"] for p in player_positions], dtype=float)

            result["teams"][tname] = {
                "player_count": len(player_positions),
                "centroid_x": round(float(t_df[x_col].mean()), 2),
                "centroid_y": round(float(t_df[y_col].mean()), 2),
                "defensive_line_x": round(float(xs.min()), 2),
                "attacking_line_x": round(float(xs.max()), 2),
                "depth": round(float(xs.max() - xs.min()), 2),
                "width": round(float(ys.max() - ys.min()), 2),
                "std_x": round(float(t_df[x_col].std(ddof=0)), 2),
                "std_y": round(float(t_df[y_col].std(ddof=0)), 2),
                "player_positions": player_positions,
            }

        return result

    # ----------------------------------------------------------------
    @staticmethod
    def _draw_pitch_canvas(
        width_px: int = 840,
        height_px: int = 544,
        margin_px: int = 36,
        draw_lines: bool = True,
        title: str = "",
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """2D saha tuvali oluşturur ve saha iç sınırlarını döndürür."""
        total_w = width_px + 2 * margin_px
        total_h = height_px + 2 * margin_px
        canvas = np.full((total_h, total_w, 3), (45, 115, 30), dtype=np.uint8)

        x0, y0 = margin_px, margin_px
        x1, y1 = margin_px + width_px, margin_px + height_px

        if draw_lines:
            sx = width_px / config.PITCH_LENGTH_M
            sy = height_px / config.PITCH_WIDTH_M

            def to_px(mx: float, my: float) -> Tuple[int, int]:
                return int(round(x0 + mx * sx)), int(round(y0 + my * sy))

            line_col = (240, 240, 240)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), line_col, 2)
            cv2.line(canvas, to_px(52.5, 0), to_px(52.5, 68), line_col, 2)
            cv2.circle(canvas, to_px(52.5, 34), int(round(9.15 * sx)), line_col, 2)
            cv2.circle(canvas, to_px(52.5, 34), 4, line_col, -1)

            # Sol ve sağ ceza sahaları + kale sahaları
            cv2.rectangle(canvas, to_px(0, 13.84), to_px(16.5, 54.16), line_col, 2)
            cv2.rectangle(canvas, to_px(88.5, 13.84), to_px(105.0, 54.16), line_col, 2)
            cv2.rectangle(canvas, to_px(0, 24.84), to_px(5.5, 43.16), line_col, 2)
            cv2.rectangle(canvas, to_px(99.5, 24.84), to_px(105.0, 43.16), line_col, 2)
        else:
            # Kalibresiz piksel alanı için çerçeve ve hafif ızgara
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 220, 200), 2)
            for frac in (0.25, 0.5, 0.75):
                gx = int(round(x0 + width_px * frac))
                gy = int(round(y0 + height_px * frac))
                cv2.line(canvas, (gx, y0), (gx, y1), (70, 135, 55), 1)
                cv2.line(canvas, (x0, gy), (x1, gy), (70, 135, 55), 1)

        if title:
            cv2.putText(
                canvas, title, (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return canvas, x0, y0, width_px, height_px

    # ----------------------------------------------------------------
    def _render_heatmap_image(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        bounds: Tuple[float, float, float, float],
        title: str,
        draw_pitch_lines: bool,
        output_path: str,
    ) -> str:
        """Verilen nokta bulutunu Gaussian yoğunluk haritası olarak PNG'ye kaydeder."""
        canvas, x0, y0, pw, ph = self._draw_pitch_canvas(
            draw_lines=draw_pitch_lines, title=title
        )
        min_x, max_x, min_y, max_y = bounds
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)

        density = np.zeros((ph, pw), dtype=np.float32)
        pxs = np.clip(((xs - min_x) / span_x) * (pw - 1), 0, pw - 1).astype(int)
        pys = np.clip(((ys - min_y) / span_y) * (ph - 1), 0, ph - 1).astype(int)
        np.add.at(density, (pys, pxs), 1.0)

        blurred = cv2.GaussianBlur(density, (0, 0), sigmaX=24, sigmaY=24)
        max_val = float(blurred.max())
        if max_val > 0:
            norm = np.clip((blurred / max_val) * 255.0, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

            roi = canvas[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
            alpha = np.where(norm > 10, (norm.astype(np.float32) / 255.0) * 0.72, 0.0)[..., None]
            blended = roi * (1.0 - alpha) + colored.astype(np.float32) * alpha
            canvas[y0:y0 + ph, x0:x0 + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    # ----------------------------------------------------------------
    def generate_heatmaps(
        self,
        output_dir: Optional[str] = None,
        top_n_players: int = 3,
    ) -> Dict[str, str]:
        """
        Takım A, Takım B ve en sık takip edilen N oyuncu için 2D ısı haritası PNG'leri üretir.
        """
        out_dir = output_dir or config.EXPORT_HEATMAP_DIR
        os.makedirs(out_dir, exist_ok=True)

        use_pitch = self.has_pitch_players
        x_col, y_col = ("pitch_x", "pitch_y") if use_pitch else ("pixel_x", "pixel_y")

        saved_paths: Dict[str, str] = {}
        if self.df.empty or x_col not in self.df.columns or y_col not in self.df.columns:
            return saved_paths

        players_df = self.df[
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (self.df["class_name"] != "referee")
            & (self.df["team"] != "Referee")
        ].dropna(subset=[x_col, y_col]).copy()

        if use_pitch and "on_pitch" in players_df.columns:
            players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        if players_df.empty:
            return saved_paths

        if use_pitch:
            bounds = (0.0, float(config.PITCH_LENGTH_M), 0.0, float(config.PITCH_WIDTH_M))
            space_suffix = "Pitch Meters"
        else:
            all_x = players_df[x_col].to_numpy(dtype=float)
            all_y = players_df[y_col].to_numpy(dtype=float)
            pad_x = max(50.0, (float(all_x.max()) - float(all_x.min())) * 0.05)
            pad_y = max(50.0, (float(all_y.max()) - float(all_y.min())) * 0.05)
            bounds = (
                max(0.0, float(all_x.min()) - pad_x),
                float(all_x.max()) + pad_x,
                max(0.0, float(all_y.min()) - pad_y),
                float(all_y.max()) + pad_y,
            )
            space_suffix = "Pixel Space (Uncalibrated)"

        for team_label, key, fname in (
            ("Team A", "team_a", "team_a_heatmap.png"),
            ("Team B", "team_b", "team_b_heatmap.png"),
        ):
            t_df = players_df[players_df["team"] == team_label]
            if not t_df.empty:
                out_path = os.path.join(out_dir, fname)
                self._render_heatmap_image(
                    xs=t_df[x_col].to_numpy(dtype=float),
                    ys=t_df[y_col].to_numpy(dtype=float),
                    bounds=bounds,
                    title=f"{team_label} Heatmap ({space_suffix})",
                    draw_pitch_lines=use_pitch,
                    output_path=out_path,
                )
                saved_paths[key] = out_path

        if top_n_players > 0:
            counts = players_df["track_id"].astype(str).value_counts().head(top_n_players)
            for tid_str in counts.index.tolist():
                p_df = players_df[players_df["track_id"].astype(str) == tid_str]
                team_tag = _clean_str(p_df["team"].dropna().iloc[0]) if p_df["team"].notna().any() else "Unknown"
                out_path = os.path.join(out_dir, f"player_{tid_str}_heatmap.png")
                self._render_heatmap_image(
                    xs=p_df[x_col].to_numpy(dtype=float),
                    ys=p_df[y_col].to_numpy(dtype=float),
                    bounds=bounds,
                    title=f"Player ID:{tid_str} ({team_tag}) - {space_suffix}",
                    draw_pitch_lines=use_pitch,
                    output_path=out_path,
                )
                saved_paths[f"player_{tid_str}"] = out_path

        return saved_paths

    # ----------------------------------------------------------------
    def export_all(
        self,
        output_dir: Optional[str] = None,
        generate_heatmaps: bool = True,
    ) -> Dict[str, Any]:
        """
        Tüm taktik analizleri çalıştırır ve JSON özeti, top kontrol zaman serisi CSV'si
        ve ısı haritası PNG'lerini diske kaydeder.
        """
        out_dir = output_dir or config.EXPORT_DIR
        os.makedirs(out_dir, exist_ok=True)

        timeline_df, possession_stats = self.compute_possession()
        pass_stats = self.detect_passes(timeline_df)
        team_shape = self.compute_team_shape()

        heatmaps: Dict[str, str] = {}
        if generate_heatmaps:
            heatmaps = self.generate_heatmaps(output_dir=os.path.join(out_dir, "heatmaps"))

        possession_csv_path = os.path.join(out_dir, config.EXPORT_POSSESSION_FILENAME)
        timeline_df.to_csv(possession_csv_path, index=False)

        summary = _to_python_scalar({
            "fps": self.fps,
            "is_calibrated": self.is_calibrated,
            "has_pitch_players": self.has_pitch_players,
            "has_pitch_ball": self.has_pitch_ball,
            "possession": possession_stats,
            "passes": pass_stats,
            "team_shape": team_shape,
            "exports": {
                "possession_timeline_csv": possession_csv_path,
                "heatmaps": heatmaps,
            },
        })

        summary_json_path = os.path.join(out_dir, config.EXPORT_ANALYTICS_SUMMARY_FILENAME)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["exports"]["analytics_summary_json"] = summary_json_path

        print(f"[MatchAnalyzer] Taktik analiz özeti kaydedildi: {summary_json_path}")
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Takip verisinden taktik maç analizi ve ısı haritası üretir.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME),
        help="Giriş takip dosyası (.csv veya .db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.EXPORT_DIR,
        help="Analiz çıktılarının kaydedileceği klasör",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS değeri (belirtilmezse veriden otomatik çıkarılır)",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Isı haritası PNG dosyalarını üretmeyi atla",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input.lower().endswith(".db"):
        analyzer = MatchAnalyzer.from_sql(args.input, fps=args.fps)
    else:
        analyzer = MatchAnalyzer.from_csv(args.input, fps=args.fps)

    summary = analyzer.export_all(
        output_dir=args.output_dir,
        generate_heatmaps=not args.no_heatmaps,
    )
    poss = summary["possession"]
    passes = summary["passes"]
    print(
        f"\n[Analiz Özeti] Mod: {poss['mode']} | "
        f"Topa Sahip Olma -> Team A: %{poss['team_a_pct']} | Team B: %{poss['team_b_pct']}"
    )
    print(
        f"[Analiz Özeti] Paslar -> Tamamlanan: {passes['total_completed_passes']} | "
        f"Top Kaybı: {passes['total_turnovers']}"
    )
