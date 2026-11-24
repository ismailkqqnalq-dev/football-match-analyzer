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

SAHANIN TAMAMI TEK KAREDE GÖRÜNMÜYORSA -- ÇOKLU KARE MODU:
    Bazı kamera açıları (ör. sabit orta saha kamerası) sadece sahanın küçük
    bir bölümünü (ör. orta yuvarlağı) net gösterir; ceza sahaları, köşeler
    hiç görünmez. Bu durumda TEK bir karede yeterli, YAYILMIŞ nokta bulamazsın
    -- ki bu, homografinin köşelere/ceza sahalarına doğru güvenilirliğini
    ciddi şekilde düşürür (kalibrasyon noktalarından uzaklaştıkça hata büyür).

    --multi-frame modu, referans kareden itibaren videoyu ileri doğru "oynatıp"
    arka planda kamera hareketini (optical flow) takip eder -- böylece
    videonun SONRAKİ bir anında (örn. bir korner sırasında ceza sahası
    netleştiğinde) durup oraya da tıklayabilirsin. Araç, o tıklamayı otomatik
    olarak referans karenin piksel koordinatına çevirir; sonunda hepsi TEK
    bir SOURCE_CALIBRATION_POINTS sözlüğünde birleşir.

    python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4 --frame-index 637 --multi-frame

    Pencere açılınca:
    - SPACE       : videoyu oynat / duraklat
    - d           : (duraklatıldığında) arabellekte ileri git; arabelleğin
                    sonundaysan bir sonraki kareyi okuyup tek adım ilerler
    - a           : (duraklatıldığında) arabellekte geri git (son ~5 saniyeyi
                    gözden geçirmek için -- kamera takibi geri sarılmaz)
    - SOL TIK     : (SADECE duraklatıldığında) o anki karedeki noktayı işaretle
    - q           : bitir, config.py formatında çıktı yazdır
    Video sahne kesmesi (farklı kameraya ani geçiş) içeriyorsa araç bunu
    otomatik tespit edip durur -- o ana kadar toplanan noktalarla devam edebilirsin.
"""

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.camera_motion_tracker import CameraMotionTracker


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


def transform_click_to_reference(px: float, py: float, cumulative_matrix: np.ndarray):
    """
    Canlı oynatımda tıklanan bir pikseli, REFERANS karenin piksel koordinat
    sistemine çevirir. cumulative_matrix, CameraMotionTracker.update()'in o
    kare için ürettiği M_t matrisidir -- frame_t pikselini referans karenin
    pikseline eşler (bkz. camera_motion_tracker.py başındaki matematik özeti).
    """
    point = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, cumulative_matrix)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def run_multiframe_calibration(video_path: str, reference_frame_index: int):
    """
    ÇOKLU KARE KALİBRASYON MODU

    Tek bir karede sahanın sadece küçük bir kısmı (ör. sadece orta yuvarlak)
    görünüyorsa, homografi o bölgede doğru olsa bile SAHANIN GENELİNDE
    (özellikle uzak köşe/ceza sahalarında) ciddi hata verebilir -- çünkü
    homografi, kalibrasyon noktalarından UZAKLAŞTIKÇA daha az güvenilir olur.

    Bu mod, videoyu REFERANS KAREDEN itibaren ileri doğru oynatıp arka planda
    kamera hareketini (camera_motion_tracker.py, optical flow) sürekli takip
    eder. Videonun SONRAKİ herhangi bir anında (örn. bir korner sırasında
    ceza sahası netleştiğinde) durup oraya da tıklayabilirsin -- araç, o
    tıklamayı OTOMATİK olarak referans karenin piksel koordinatına çevirir.
    Sonunda hepsi TEK bir SOURCE_CALIBRATION_POINTS sözlüğünde, sahanın çok
    daha geniş bir bölümünü kapsayacak şekilde birleşir.

    SINIRLAMA: Takip SADECE İLERİ yönde çalışır (optical flow'un doğası
    gereği). Videoda SAHNE KESMESİ (farklı kameraya ani geçiş, ör. tekrar
    izleme) olursa oradan sonrası referans kareyle bağlantısını kaybeder --
    araç bunu tespit edip otomatik durur, o ana kadar toplanan noktalarla
    devam edebilirsin.

    NOT: Bu araçta oyuncu/top maskeleme (YOLO) ÇALIŞTIRILMIYOR -- optical
    flow, kareler arasında oyuncular dahil HER YERDE özellik noktası arayabilir.
    Ana pipeline'daki (main.py) kadar hassas değildir ama kalibrasyon
    amacıyla (birkaç sabit nokta bulmak) yeterince sağlamdır. Kalabalık/hızlı
    hareketli anlarda tıklamadan önce videoyu biraz beklet, daha sakin bir
    anda işaretle.

    Tuşlar:
        SPACE : oynat / duraklat
        d     : ileri (duraklatıldığında: arabellekte ileri kaydır ya da
                canlı uçtaysa yeni kare oku; oynatılırken: hız değişmez)
        a     : geri (SADECE arabellekteki son ~5 saniyeyi gözden geçirmek
                için -- kamera takibi geri sarılmaz, sadece daha önce
                görülmüş kareleri tekrar gösterir)
        SOL TIK (sadece duraklatıldığında): o anki karedeki noktayı işaretle
        q     : bitir
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    buffer_size = max(30, int(fps * 5))  # ~5 saniyelik "geri gözden geçirme" arabelleği

    ref_frame = read_frame_at(cap, reference_frame_index)
    if ref_frame is None:
        raise IOError(f"Referans kare okunamadı (frame_idx={reference_frame_index}): {video_path}")

    motion_tracker = CameraMotionTracker()
    ref_matrix = motion_tracker.update(ref_frame, [])  # ilk çağrı -> birim matris (referansın kendisi)

    # Arabellek: her eleman (frame_idx, frame_görüntüsü, o karenin_referansa_göre_matrisi)
    buffer = deque(maxlen=buffer_size)
    buffer.append((reference_frame_index, ref_frame, ref_matrix))
    buf_pos = 0  # arabellek içindeki "şu an gösterilen" konum (0 = en eski, len-1 = en yeni/canlı)

    next_read_index = reference_frame_index + 1
    playing = False
    scene_cut_hit = False

    # collected_points: {"ref_x", "ref_y", "source_frame_idx", "raw_x", "raw_y"}
    collected_points = []

    window_name = "Coklu Kare Kalibrasyon - SPACE: oynat/duraklat, a/d: gezin, SOL TIK: nokta, q: bitir"
    cv2.namedWindow(window_name)

    def current_entry():
        return buffer[buf_pos]

    def redraw():
        frame_idx, frame_img, _matrix = current_entry()
        display = frame_img.copy()

        # Bu KAREDE (frame_idx) daha önce işaretlenmiş noktaları tekrar çiz
        for p in collected_points:
            if p["source_frame_idx"] == frame_idx:
                x, y = int(p["raw_x"]), int(p["raw_y"])
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        status = "OYNATILIYOR" if playing else "DURAKLATILDI"
        cut_note = "  |  [SAHNE KESMESİ TESPİT EDİLDİ -- daha fazla ilerlenemez]" if scene_cut_hit else ""
        info = (
            f"Kare: {frame_idx} (referans: {reference_frame_index})  |  "
            f"{status}  |  Toplam nokta: {len(collected_points)}{cut_note}"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(display, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or playing:
            return  # sadece duraklatıldığında tıklama kabul edilir
        frame_idx, _frame_img, matrix = current_entry()
        ref_x, ref_y = transform_click_to_reference(x, y, matrix)
        collected_points.append({
            "ref_x": ref_x, "ref_y": ref_y,
            "source_frame_idx": frame_idx, "raw_x": x, "raw_y": y,
        })
        print(
            f"[{len(collected_points)}] Kare {frame_idx}'te tıklandı (x={x}, y={y}) "
            f"-> referans kareye çevrildi: (x={ref_x:.1f}, y={ref_y:.1f})"
        )
        redraw()

    cv2.setMouseCallback(window_name, on_click)
    redraw()

    print("\n" + "=" * 70)
    print("ÇOKLU KARE KALİBRASYON BAŞLADI")
    print(f"Referans kare: {reference_frame_index}  |  Video: {video_path}")
    print("SPACE: oynat/duraklat  |  a/d: arabellekte gezin  |  SOL TIK (duraklatıldığında): nokta işaretle")
    print("Sahanın farklı bölgelerinin (ceza sahası, köşe, taç çizgisi) göründüğü anları bekle,")
    print("SPACE ile duraklat, tıkla, tekrar SPACE ile devam et. Bitirince 'q'.")
    print("=" * 70 + "\n")

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
            if playing:
                buf_pos = len(buffer) - 1  # oynatmaya devam ederken hep canlı uçtan başla
            redraw()
        elif key == ord("a") and not playing:
            buf_pos = max(0, buf_pos - 1)
            redraw()
        elif key == ord("d") and not playing:
            if buf_pos < len(buffer) - 1:
                buf_pos += 1
                redraw()
            # canlı uçtaysak, 'd' bir sonraki kareyi OKUYUP takip eder (aşağıdaki playing bloğuyla ortak mantık)
            elif not scene_cut_hit and next_read_index < total_frames:
                playing = True  # bir adım ilerlet, aşağıdaki blok işleyecek

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # --- Canlı ilerleme: sadece "oynatılıyor" durumundayken yeni kare oku ---
        if playing and not scene_cut_hit:
            if next_read_index >= total_frames:
                print("\nVideo sonuna ulaşıldı, oynatma durduruluyor.")
                playing = False
                redraw()
                continue

            success, frame = cap.read()
            if not success:
                print("\nKare okunamadı, oynatma durduruluyor.")
                playing = False
                redraw()
                continue

            matrix = motion_tracker.update(frame, [])
            if matrix is None:
                print(
                    "\n[UYARI] SAHNE KESMESİ tespit edildi -- kamera takibi referans kareyle "
                    "bağlantısını kaybetti. Buradan sonrası için nokta eklenemez. Şu ana kadar "
                    "toplanan noktalarla devam edebilirsin ('q' ile bitir)."
                )
                scene_cut_hit = True
                playing = False
                redraw()
                continue

            buffer.append((next_read_index, frame, matrix))
            buf_pos = len(buffer) - 1
            next_read_index += 1
            # Tek adım "d" ile gelindiyse otomatik tekrar duraklat
            if key == ord("d"):
                playing = False
            redraw()

    cap.release()
    cv2.destroyAllWindows()

    if not collected_points:
        print("Hiç nokta tıklanmadı, çıkılıyor.")
        return

    # --- Kullanıcıdan her tıklanan nokta için hangi landmark'a karşılık geldiğini iste ---
    print(f"\nReferans kare: {reference_frame_index}. Toplam {len(collected_points)} nokta toplandı "
          f"(farklı karelerden, hepsi referans kareye çevrildi).")
    print("Şimdi her nokta için hangi saha noktasına karşılık geldiğini gir.")
    print(f"Geçerli isimler: {list(config.PITCH_LANDMARKS_METERS.keys())}\n")

    result = {}
    for idx, p in enumerate(collected_points, start=1):
        landmark_name = input(
            f"Nokta {idx} (kare {p['source_frame_idx']}'ten, referansa göre "
            f"x={p['ref_x']:.1f}, y={p['ref_y']:.1f}) -> landmark ismi: "
        ).strip()
        if landmark_name in config.PITCH_LANDMARKS_METERS:
            result[landmark_name] = (round(p["ref_x"], 1), round(p["ref_y"], 1))
        else:
            print(f"  UYARI: '{landmark_name}' tanınmıyor, bu nokta atlanacak.")

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
    print(f"CALIBRATION_FRAME_INDEX = {reference_frame_index}")
    print("=" * 70)


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


# ----------------------------------------------------------------------
# A2-6: İsabet ölçümü için çok kareli etiketleme modu (--label-mode)
# ----------------------------------------------------------------------
LABEL_COLUMNS = ("frame_idx", "landmark_name", "pixel_x", "pixel_y")


def resolve_label_frames(total_frames: int, fps: float, frames_arg=None, every_sec=None):
    """--frames "100,350" ya da --every-sec 15 değerinden etiketlenecek kare listesini üretir."""
    total_frames = max(1, int(total_frames))
    if frames_arg:
        frames = sorted({int(x) for x in str(frames_arg).split(",") if x.strip() != ""})
    elif every_sec:
        step = max(1, int(round(float(every_sec) * (fps or 25.0))))
        frames = list(range(0, total_frames, step))
    else:
        raise ValueError("Etiketleme modu için --frames ya da --every-sec gerekli.")
    bad = [f for f in frames if f < 0 or f >= total_frames]
    if bad:
        raise ValueError(f"Video aralığı dışında kareler: {bad} (toplam {total_frames} kare)")
    return frames


def label_landmark_names():
    from utils.evaluate_homography import label_landmarks_meters

    return sorted(label_landmarks_meters().keys())


def parse_landmark_answer(answer: str, names):
    """Kullanıcı yanıtını (isim ya da listedeki numara) saha noktası adına çevirir; boş/geçersizse None."""
    answer = (answer or "").strip()
    if not answer:
        return None
    if answer.isdigit():
        idx = int(answer)
        return names[idx] if 0 <= idx < len(names) else None
    return answer if answer in names else None


def write_labels_csv(path: str, rows) -> str:
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(LABEL_COLUMNS)
        for r in rows:
            w.writerow([int(r[0]), r[1], round(float(r[2]), 1), round(float(r[3]), 1)])
    return path


def run_label_mode(video_path: str, frames, output_path: str):
    """
    Seçilen her karede bilinen saha noktalarına SOL TIKLA; 'n' ile sonraki kareye geç,
    'q' ile bitir. Her karenin sonunda tıklanan noktaların adları terminalden sorulur
    (isim ya da listedeki numara; boş bırakılan nokta atlanır).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")
    names = label_landmark_names()
    print("Geçerli saha noktaları (numara ya da isim yazabilirsin):")
    for i, n in enumerate(names):
        print(f"  {i:2d}: {n}")

    rows = []
    window_name = "Etiketleme - SOL TIK: nokta, n: sonraki kare, q: bitir"
    cv2.namedWindow(window_name)
    stop = False
    for k, f in enumerate(frames, start=1):
        frame = read_frame_at(cap, f)
        if frame is None:
            print(f"[UYARI] Kare {f} okunamadı, atlanıyor.")
            continue
        clicked_points.clear()

        def redraw(frame=frame, f=f, k=k):
            disp = frame.copy()
            for i, (x, y) in enumerate(clicked_points, start=1):
                cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(disp, f"Kare {f} ({k}/{len(frames)}) | {len(clicked_points)} nokta",
                        (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(window_name, disp)

        cv2.setMouseCallback(window_name, mouse_callback, param={"redraw": redraw})
        redraw()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("n"), ord("q")):
                stop = key == ord("q")
                break
        for i, (x, y) in enumerate(list(clicked_points), start=1):
            name = parse_landmark_answer(input(f"Kare {f} nokta {i} (x={x}, y={y}) -> saha noktası: "), names)
            if name is None:
                print("  atlandı")
                continue
            rows.append((f, name, x, y))
        if stop:
            break
    cap.release()
    cv2.destroyAllWindows()
    write_labels_csv(output_path, rows)
    print(f"{len(rows)} etiket yazıldı: {output_path}")
    print("Ölçüm: python utils/evaluate_homography.py --log <export>/homography_log.csv "
          f"--labels {output_path}")


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
    parser.add_argument(
        "--multi-frame", action="store_true",
        help=(
            "Referans kareden ileri doğru oynatarak, videonun FARKLI ANLARINDAN "
            "nokta toplamayı sağlayan gelişmiş mod (bkz. run_multiframe_calibration "
            "docstring'i). Sahanın tamamı tek bir karede görünmüyorsa kullan."
        ),
    )
    parser.add_argument(
        "--label-mode", action="store_true",
        help="A2-6: isabet ölçümü için birden fazla karede saha noktası etiketle (labels.csv)",
    )
    parser.add_argument("--frames", type=str, default=None, help='Etiketlenecek kareler, örn. "100,350,900"')
    parser.add_argument("--every-sec", type=float, default=None, help="Her N saniyede bir kare etiketle")
    parser.add_argument(
        "--labels-out", type=str, default=os.path.join(config.EXPORT_DIR, "labels.csv"),
        help="Etiket dosyası yolu (varsayılan: data/exports/labels.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.label_mode:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {args.input}")
        label_frames = resolve_label_frames(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25.0,
            args.frames, args.every_sec,
        )
        cap.release()
        run_label_mode(args.input, label_frames, args.labels_out)
    elif args.save_frame_only:
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

            if args.multi_frame:
                run_multiframe_calibration(args.input, start_index)
            else:
                run_interactive_calibration(args.input, start_index)
        except cv2.error as e:
            print(f"\nHATA: Ekran açılamadı ({e}).")
            print("Muhtemelen ekranı olmayan (headless) bir ortamdasın.")
            print("Bunun yerine şunu dene: python utils/calibrate_homography.py --save-frame-only")
