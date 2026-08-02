"""
utils/calibrate_homography.py
------------------------------
YARDIMCI ARAÇ: Homografi kalibrasyonu için piksel koordinatı toplama.

NEDEN BU ARAÇ GEREKLİ?
    perspective_transformer.py'nin çalışabilmesi için, config.py'deki
    SOURCE_CALIBRATION_POINTS sözlüğünü DOLDURMAMIZ gerekiyor -- yani sahadaki
    bilinen noktaların (köşe bayrağı, ceza sahası köşesi vb.) SENİN VİDEONDAKİ
    piksel karşılıklarını bulmamız lazım. Bunu elle, video karesine bakıp
    tahmin ederek yapmak hem zahmetli hem hataya açık.

    Bu araç videonun BİR KARESİNİ bir pencerede açar. Sen sahadaki bilinen bir
    noktaya (örn. sol üst köşe bayrağı) tıklarsın, program o tıklamanın piksel
    koordinatını (x, y) konsola ve ekrana yazar. Sırayla birkaç nokta
    tıklarsın, sonunda çıktıyı doğrudan config.py'ye yapıştırırsın.

HANGİ KAREDE KALİBRASYON YAPMALIYIM? -- VİDEONUN İLK KARESİ ZORUNLU DEĞİL!
    Video başında saha net görünmüyorsa (intro, tekrar izleme, farklı bir açı,
    kapanış logosu vb.), sahanın EN NET, EN GENİŞ göründüğü herhangi bir
    kareyi seçebilirsin. Pencere açıldıktan sonra klavye ile kare kare (ya da
    ~1 saniyelik atlamalarla) gezinip doğru kareyi bulabilirsin -- video
    numarasını önceden bilmene gerek yok (bkz. aşağıdaki tuş kısayolları).

    ÖNEMLİ: Hangi kareyi seçersen seç, işlem bitince ekrana basılan
    "CALIBRATION_FRAME_INDEX = ..." satırını da config.py'ye kopyalaman
    GEREKİYOR -- main.py, kamera hareketi telafisini (Adım 5 eklentisi) TAM
    OLARAK bu kareden başlatarak kalibrasyonla hizalıyor. Bu adımı atlarsan
    saha koordinatları yanlış çıkar.

ÖNEMLİ NOT (headless / sunucu ortamları için):
    Bu araç cv2.imshow() kullandığı için EKRANI OLAN bir ortamda (kendi
    bilgisayarın, masaüstü) çalıştırılmalıdır. SSH ile bağlanılan headless
    bir sunucuda veya bu sohbetin çalıştığı konteyner gibi ekranı olmayan
    ortamlarda cv2.imshow() hata verir. Bu durumda alternatif olarak istediğin
    kareyi bir PNG olarak kaydedip (bkz. save_frame_as_image fonksiyonu),
    herhangi bir resim düzenleme programında (Paint, Photoshop, GIMP,
    hatta çevrimiçi bir "pixel coordinate finder" aracı) piksel koordinatlarını
    okuyabilirsin.

KULLANIM (ekranlı ortamda, önerilen -- video başındaki bir kareden gezinerek başla):
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4

    Belirli bir kareden/saniyeden başlamak istersen:
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4 --frame-index 350
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4 --timestamp 14.5

    Pencere açılınca:
    - 'd' / 'a'  : bir sonraki / bir önceki kareye geç
    - 'w' / 's'  : ~1 saniye ileri / geri atla (hızlı gezinme için)
    - Sahadaki bilinen bir noktaya SOL TIKLA (not: kare değiştirirsen, önceki
      karede tıkladığın noktalar TEMİZLENİR -- tüm noktalar AYNI karede
      işaretlenmiş olmalı, farklı karelerden karışık nokta toplanamaz)
    - 'q' tuşuna basınca çıkar; terminalde her nokta için hangi landmark
      olduğunu gireceksin, sonunda config.py formatında çıktı yazdırılır

KULLANIM (headless / ekransız ortamda):
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4 --frame-index 350 --save-frame-only
    (Bu, sadece seçtiğin kareyi data/exports/calibration_frame.png olarak
     kaydeder; koordinatları başka bir araçla bulman gerekir.)
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# Tıklanan noktaları saklamak için global liste (cv2 mouse callback'i başka
# şekilde veri döndüremediği için bu yaygın bir OpenCV pattern'idir).
clicked_points = []


def read_frame_at(cap: cv2.VideoCapture, frame_index: int):
    """
    Bir video yakalama nesnesinden BELİRLİ bir kareyi okur.

    NOT: cv2.set(cv2.CAP_PROP_POS_FRAMES, ...) bazı video codec'lerinde
    (özellikle değişken kare hızlı / VFR dosyalarda) birkaç kare sapma
    gösterebilir -- bu OpenCV'nin bilinen bir sınırlamasıdır. Pencere
    içindeki kare sayacı her zaman GERÇEKTEN OKUNAN kareyi gösterdiği için,
    ufak bir sapma olsa bile ne gördüğünü bilerek işaretleme yaparsın.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    return frame if success else None


def resolve_start_frame_index(cap: cv2.VideoCapture, args) -> int:
    """--frame-index / --timestamp / config varsayılanından başlangıç karesini belirler."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.timestamp is not None:
        return max(0, int(round(args.timestamp * fps)))
    if args.frame_index is not None:
        return max(0, args.frame_index)
    return max(0, config.CALIBRATION_FRAME_INDEX)


def mouse_callback(event, x, y, flags, param):
    """Sol tık algılandığında piksel koordinatını kaydeder."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[{len(clicked_points)}] Piksel koordinatı kaydedildi: (x={x}, y={y})")
        param["redraw"]()


def save_frame_as_image(video_path: str, frame_index: int, output_path: str):
    """Headless ortamlar için: videonun BELİRTİLEN karesini bir PNG dosyası olarak kaydeder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")

    frame = read_frame_at(cap, frame_index)
    cap.release()

    if frame is None:
        raise IOError(f"Kare okunamadı (frame_idx={frame_index}): {video_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Kare (frame_idx={frame_index}) kaydedildi: {output_path}")
    print(
        "Şimdi bu görseli bir resim düzenleyicide açıp, sahadaki bilinen "
        "noktaların piksel koordinatlarını (x, y) okuyabilir ve bunları "
        "config.py -> SOURCE_CALIBRATION_POINTS içine elle girebilirsin.\n"
        f"Unutma: config.py -> CALIBRATION_FRAME_INDEX = {frame_index} olarak da ayarlamalısın."
    )


def run_interactive_calibration(video_path: str, start_frame_index: int):
    """Ekranlı ortamlarda: kare gezinme + tıklama tabanlı interaktif kalibrasyon penceresi."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    one_second_jump = max(1, int(round(fps)))

    state = {"index": max(0, min(start_frame_index, total_frames - 1)), "frame": None}
    state["frame"] = read_frame_at(cap, state["index"])
    if state["frame"] is None:
        raise IOError(f"Kare okunamadı (frame_idx={state['index']}): {video_path}")

    window_name = "Kalibrasyon - a/d: kare, w/s: ~1sn atla, SOL TIK: nokta, q: bitir"
    cv2.namedWindow(window_name)

    def redraw():
        """Mevcut kareyi + üzerine şu ana kadar tıklanan noktaları + bilgi satırını çizer."""
        display = state["frame"].copy()
        for i, (x, y) in enumerate(clicked_points, start=1):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        info = f"Kare: {state['index']}/{total_frames - 1}  (t={state['index'] / fps:.2f}s)  |  {len(clicked_points)} nokta"
        cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(display, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

    def go_to_frame(new_index: int):
        new_index = max(0, min(new_index, total_frames - 1))
        if new_index == state["index"]:
            return
        frame = read_frame_at(cap, new_index)
        if frame is None:
            print(f"[UYARI] Kare {new_index} okunamadı, mevcut karede kalınıyor.")
            return
        if clicked_points:
            print(
                f"[UYARI] Kare değiştirildi ({state['index']} -> {new_index}) -- önceki karede "
                f"tıklanan {len(clicked_points)} nokta TEMİZLENDİ (tüm noktalar aynı kareden olmalı)."
            )
            clicked_points.clear()
        state["index"] = new_index
        state["frame"] = frame
        redraw()

    cv2.setMouseCallback(window_name, mouse_callback, param={"redraw": redraw})
    redraw()

    print("\n" + "=" * 70)
    print("KALİBRASYON BAŞLADI")
    print(f"Video: {video_path}  |  Toplam kare: {total_frames}  |  FPS: {fps:.2f}")
    print("Tuşlar:  d = ileri 1 kare   a = geri 1 kare")
    print("         w = ileri ~1 saniye   s = geri ~1 saniye")
    print("Sahadaki bilinen bir noktaya (örn. köşe bayrağı, ceza sahası köşesi) SOL TIKLA.")
    print("Mevcut landmark isimleri için config.py -> PITCH_LANDMARKS_METERS'a bak.")
    print("Sahanın net göründüğü doğru kareyi bulunca işaretlemeye başla. Bitirmek için 'q'.")
    print("=" * 70 + "\n")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            go_to_frame(state["index"] + 1)
        elif key == ord("a"):
            go_to_frame(state["index"] - 1)
        elif key == ord("w"):
            go_to_frame(state["index"] + one_second_jump)
        elif key == ord("s"):
            go_to_frame(state["index"] - one_second_jump)
        # Pencere kapatma butonuna basıldıysa da çık (bazı platformlarda 'q' algılanmayabilir)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    final_frame_index = state["index"]
    cap.release()
    cv2.destroyAllWindows()

    if not clicked_points:
        print("Hiç nokta tıklanmadı, çıkılıyor.")
        return

    # --- Kullanıcıdan her tıklanan nokta için hangi landmark'a karşılık geldiğini iste ---
    print(f"\nKalibrasyon karesi: frame_idx={final_frame_index} (t={final_frame_index / fps:.2f}s)")
    print("Şimdi her tıkladığın nokta için hangi saha noktasına karşılık geldiğini gir.")
    print(f"Geçerli isimler: {list(config.PITCH_LANDMARKS_METERS.keys())}\n")

    result = {}
    for idx, (x, y) in enumerate(clicked_points, start=1):
        landmark_name = input(f"Nokta {idx} (piksel x={x}, y={y}) -> landmark ismi: ").strip()
        if landmark_name in config.PITCH_LANDMARKS_METERS:
            result[landmark_name] = (x, y)
        else:
            print(f"  UYARI: '{landmark_name}' tanınmıyor, bu nokta atlanacak.")

    # --- config.py'ye kopyalanabilir formatta çıktı ver ---
    print("\n" + "=" * 70)
    print("Aşağıdaki İKİ bloğu doğrudan config.py'ye yapıştır:")
    print("=" * 70)
    print("1) SOURCE_CALIBRATION_POINTS'i şununla DEĞİŞTİR:")
    print("SOURCE_CALIBRATION_POINTS = {")
    for name, (x, y) in result.items():
        print(f'    "{name}": ({x}, {y}),')
    print("}")
    print()
    print("2) CALIBRATION_FRAME_INDEX'i şununla DEĞİŞTİR (ÖNEMLİ, unutma!):")
    print(f"CALIBRATION_FRAME_INDEX = {final_frame_index}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Homografi kalibrasyon aracı")
    parser.add_argument("--input", type=str, default=config.DEFAULT_INPUT_VIDEO)
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="Kalibrasyona hangi kareden başlanacağı (varsayılan: config.CALIBRATION_FRAME_INDEX). "
             "Pencere açıldıktan sonra a/d/w/s ile yine de gezinebilirsin.",
    )
    parser.add_argument(
        "--timestamp", type=float, default=None,
        help="Kalibrasyona hangi saniyeden başlanacağı (--frame-index yerine kullanılabilir).",
    )
    parser.add_argument(
        "--save-frame-only", action="store_true",
        help="Ekranı olmayan ortamlar için: sadece seçilen kareyi PNG olarak kaydet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save_frame_only:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {args.input}")
        frame_index = resolve_start_frame_index(cap, args)
        cap.release()

        output_path = os.path.join(config.EXPORT_DIR, "calibration_frame.png")
        save_frame_as_image(args.input, frame_index, output_path)
    else:
        try:
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                raise IOError(f"Video açılamadı: {args.input}")
            start_index = resolve_start_frame_index(cap, args)
            cap.release()

            run_interactive_calibration(args.input, start_index)
        except cv2.error as e:
            print(f"\nHATA: Ekran açılamadı ({e}).")
            print("Muhtemelen ekranı olmayan (headless) bir ortamdasın.")
            print("Bunun yerine şunu dene: python utils/calibrate_homography.py --save-frame-only")
