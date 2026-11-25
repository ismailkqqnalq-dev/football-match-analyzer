"""
label_events.py
---------------
Faz V: Videoyu izlerken pas, top kaybı ve top sahipliğini tuşla işaretleme aracı.
Çıktıları utils/validate_events.py doğrudan okur.

Tuşlar:
  boşluk   oynat / duraklat
  1 / 2    top artık Takım A / Takım B'de (sahiplik başlar)
  0        top kimsede değil (taç, aut, havada, oyun durdu)
  p        tamamlanan pas (topun ayaktan çıktığı an; takım = o anki sahip takım)
  t        top kaybı (kaybeden takımın topu ayağından çıkardığı an)
  u        son işareti geri al
  a / d    bir kare geri / ileri (duraklatılmışken)
  j / l    5 sn geri / ileri
  [ / ]    yavaşlat / hızlandır
  q / Esc  kaydet ve çık

Takım A / B adları sistemin adlarıyla aynı olmak zorunda değil; doğrulama aracı
gerekirse ikisini yer değiştirerek eşleştirir.

Kullanım:
    python utils/label_events.py --video klip.mp4 --out-dir etiketler/ [--start-sec 0 --end-sec 300]
"""

import argparse
import csv
import os
from typing import List, Optional, Tuple

TEAMS = {ord("1"): "Team A", ord("2"): "Team B", ord("0"): None}
EVENTS = {ord("p"): "completed_pass", ord("t"): "turnover"}


class LabelSession:
    """İşaretleri tutar; video ve pencereden bağımsızdır (test edilebilir)."""

    def __init__(self) -> None:
        self.marks: List[Tuple[float, str, Optional[str]]] = []  # (time_sec, kind, team)

    def team_at(self, t: float) -> Optional[str]:
        team = None
        for time_sec, kind, tm in sorted(self.marks, key=lambda m: m[0]):
            if time_sec > t:
                break
            if kind == "possession":
                team = tm
        return team

    def set_possession(self, t: float, team: Optional[str]) -> None:
        self.marks.append((t, "possession", team))

    def add_event(self, t: float, event_type: str) -> None:
        self.marks.append((t, event_type, self.team_at(t)))

    def undo(self) -> Optional[Tuple[float, str, Optional[str]]]:
        return self.marks.pop() if self.marks else None

    def events(self) -> List[Tuple[float, str, Optional[str]]]:
        return sorted(m for m in self.marks if m[1] != "possession")

    def possession_segments(self, end_sec: float) -> List[Tuple[float, float, Optional[str]]]:
        pos = sorted((m for m in self.marks if m[1] == "possession"), key=lambda m: m[0])
        segs = []
        for i, (t, _, team) in enumerate(pos):
            t_end = pos[i + 1][0] if i + 1 < len(pos) else end_sec
            if t_end > t:
                segs.append((t, t_end, team))
        return segs

    def save(self, out_dir: str, end_sec: float) -> Tuple[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        ev_path = os.path.join(out_dir, "gt_events.csv")
        pos_path = os.path.join(out_dir, "gt_possession.csv")
        with open(ev_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "event_type", "team"])
            for t, kind, team in self.events():
                w.writerow([round(t, 3), kind, team or ""])
        with open(pos_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sec", "end_sec", "team"])
            for s, e, team in self.possession_segments(end_sec):
                w.writerow([round(s, 3), round(e, 3), team or ""])
        return ev_path, pos_path


def run(video: str, out_dir: str, start_sec: float = 0.0, end_sec: Optional[float] = None) -> None:
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_sec * fps)
    last = min(n - 1, int(end_sec * fps)) if end_sec is not None else n - 1
    sess = LabelSession()
    frame_idx, playing, speed = first, False, 1.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    win = "etiketle (q: kaydet ve cik)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def seek(idx: int):
        nonlocal frame_idx, frame
        frame_idx = max(first, min(last, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()

    while ok and frame is not None:
        t = frame_idx / fps
        disp = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))
        team = sess.team_at(t)
        recent = [m for m in sess.events() if 0 <= t - m[0] < 1.0]
        info = f"{t:7.2f}s  x{speed:.2f}  {'>' if playing else '||'}  top: {team or '-'}  olay: {len(sess.events())}"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(disp, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if recent:
            cv2.putText(disp, recent[-1][1].upper(), (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(win, disp)

        key = cv2.waitKey(max(1, int(1000 / (fps * speed))) if playing else 0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key in TEAMS:
            sess.set_possession(t, TEAMS[key])
        elif key in EVENTS:
            sess.add_event(t, EVENTS[key])
            sess.save(out_dir, last / fps)
        elif key == ord("u"):
            sess.undo()
        elif key == ord("a"):
            seek(frame_idx - 1)
            continue
        elif key == ord("d"):
            seek(frame_idx + 1)
            continue
        elif key == ord("j"):
            seek(frame_idx - int(5 * fps))
            continue
        elif key == ord("l"):
            seek(frame_idx + int(5 * fps))
            continue
        elif key == ord("["):
            speed = max(0.125, speed / 2)
        elif key == ord("]"):
            speed = min(4.0, speed * 2)

        if playing:
            if frame_idx >= last:
                playing = False
                continue
            ok, frame = cap.read()
            frame_idx += 1

    paths = sess.save(out_dir, last / fps)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Kaydedildi: {paths[0]} ({len(sess.events())} olay), {paths[1]}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Faz V olay etiketleme aracı")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float)
    args = ap.parse_args(argv)
    run(args.video, args.out_dir, args.start_sec, args.end_sec)


if __name__ == "__main__":
    main()
