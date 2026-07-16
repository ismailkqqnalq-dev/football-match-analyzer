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

    Bu araç, videonun ilk karesini bir pencerede açar. Sen sahadaki bilinen
    bir noktaya (örn. sol üst köşe bayrağı) tıklarsın, program o tıklamanın
    piksel koordinatını (x, y) konsola ve ekrana yazar. Sırayla birkaç nokta
    tıklarsın, sonunda çıktıyı doğrudan config.py'ye yapıştırırsın.

ÖNEMLİ NOT (headless / sunucu ortamları için):
    Bu araç cv2.imshow() kullandığı için EKRANI OLAN bir ortamda (kendi
    bilgisayarın, masaüstü) çalıştırılmalıdır. SSH ile bağlanılan headless
    bir sunucuda veya bu sohbetin çalıştığı konteyner gibi ekranı olmayan
    ortamlarda cv2.imshow() hata verir. Bu durumda alternatif olarak ilk
    kareyi bir PNG olarak kaydedip (bkz. save_first_frame_as_image fonksiyonu),
    herhangi bir resim düzenleme programında (Paint, Photoshop, GIMP,
    hatta çevrimiçi bir "pixel coordinate finder" aracı) piksel koordinatlarını
    okuyabilirsin.

KULLANIM (ekranlı ortamda):
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4

    Pencere açılınca:
    - Sahadaki bilinen bir noktaya SOL TIKLA
    - Terminalde o an hangi landmark ismini işaretlediğini gireceksin
    - 'q' tuşuna basınca çıkar ve topladığın tüm noktaları config.py formatında yazdırır

KULLANIM (headless / ekransız ortamda):
    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4 --save-frame-only
    (Bu, sadece ilk kareyi data/exports/first_frame.png olarak kaydeder;
     koordinatları başka bir araçla bulman gerekir.)
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


def mouse_callback(event, x, y, flags, param):
    """Sol tık algılandığında piksel koordinatını kaydeder ve ekrana çizer."""
    frame = param["frame"]

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # Tıklanan noktayı görsel olarak işaretle (kırmızı nokta + sıra numarası)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame, str(len(clicked_points)), (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow("Kalibrasyon - Sahadaki bilinen noktalara tikla, 'q' ile bitir", frame)

        print(f"[{len(clicked_points)}] Piksel koordinatı kaydedildi: (x={x}, y={y})")


def save_first_frame_as_image(video_path: str, output_path: str):
    """Headless ortamlar için: videonun ilk karesini bir PNG dosyası olarak kaydeder."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise IOError(f"Videonun ilk karesi okunamadı: {video_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"İlk kare kaydedildi: {output_path}")
    print(
        "Şimdi bu görseli bir resim düzenleyicide açıp, sahadaki bilinen "
        "noktaların piksel koordinatlarını (x, y) okuyabilir ve bunları "
        "config.py -> SOURCE_CALIBRATION_POINTS içine elle girebilirsin."
    )


def run_interactive_calibration(video_path: str):
    """Ekranlı ortamlarda: tıklama tabanlı interaktif kalibrasyon penceresini açar."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise IOError(f"Videonun ilk karesi okunamadı: {video_path}")

    window_name = "Kalibrasyon - Sahadaki bilinen noktalara tikla, 'q' ile bitir"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param={"frame": frame})
    cv2.imshow(window_name, frame)

    print("\n" + "=" * 70)
    print("KALİBRASYON BAŞLADI")
    print("Sahadaki bilinen bir noktaya (örn. köşe bayrağı, ceza sahası köşesi) tıkla.")
    print("Mevcut landmark isimleri için config.py -> PITCH_LANDMARKS_METERS'a bak.")
    print("Bitirmek için 'q' tuşuna bas.")
    print("=" * 70 + "\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not clicked_points:
        print("Hiç nokta tıklanmadı, çıkılıyor.")
        return

    # --- Kullanıcıdan her tıklanan nokta için hangi landmark'a karşılık geldiğini iste ---
    print("\nŞimdi her tıkladığın nokta için hangi saha noktasına karşılık geldiğini gir.")
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
    print("Aşağıdaki bloğu doğrudan config.py -> SOURCE_CALIBRATION_POINTS içine yapıştır:")
    print("=" * 70)
    print("SOURCE_CALIBRATION_POINTS = {")
    for name, (x, y) in result.items():
        print(f'    "{name}": ({x}, {y}),')
    print("}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Homografi kalibrasyon aracı")
    parser.add_argument("--input", type=str, default=config.DEFAULT_INPUT_VIDEO)
    parser.add_argument(
        "--save-frame-only", action="store_true",
        help="Ekranı olmayan ortamlar için: sadece ilk kareyi PNG olarak kaydet"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save_frame_only:
        output_path = os.path.join(config.EXPORT_DIR, "first_frame.png")
        save_first_frame_as_image(args.input, output_path)
    else:
        try:
            run_interactive_calibration(args.input)
        except cv2.error as e:
            print(f"\nHATA: Ekran açılamadı ({e}).")
            print("Muhtemelen ekranı olmayan (headless) bir ortamdasın.")
            print("Bunun yerine şunu dene: python utils/calibrate_homography.py --save-frame-only")
