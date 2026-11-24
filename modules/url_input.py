"""
url_input.py
------------
`main.py --input` parametresine dosya yolu yerine bir video/maç linki
(http/https) verilebilmesini sağlar. Desteklenen link türleri:
  * video paylaşım siteleri (ör. https://www.youtube.com/watch?v=...),
  * içinde gömülü video olan sayfalar (haber, kulüp, maç siteleri),
  * doğrudan video dosyası linkleri (.mp4, .webm, .mkv),
  * HLS / DASH akış linkleri (.m3u8, .mpd).
Hepsi tek yoldan, yt-dlp ile çözülür (doğrudan dosya ve akış linklerini de
yt-dlp'nin genel "generic" çözücüsü tanır), böylece ayrı indirme kodu gerekmez.

Yaklaşım: video bir kez işletim sisteminin geçici klasörüne
(`%TEMP%/fma_url_cache`) indirilir; aynı link tekrar verilirse önbellekteki
dosya kullanılır. Doğrudan akış (stream) adresini cv2.VideoCapture'a vermek
yerine indirme seçildi, çünkü:
  * VideoProcessor yerel dosya bekliyor (os.path.exists kontrolü),
  * akışta FPS / toplam kare sayısı bilinmeyebilir; FPS zaman damgası, hız
    ve mesafe hesaplarının temelidir,
  * çoğu sitenin akış adresi birkaç saat içinde geçersiz olur.
"""

import hashlib
import os
import tempfile

# En fazla 1080p, tercihen mp4/h264 (avc1). Yalnızca görüntü akışı yeterli
# (ses kullanılmıyor); böylece ses+görüntü birleştirmek için ffmpeg gerekmez.
YTDLP_FORMAT = (
    "bv*[height<=1080][ext=mp4][vcodec^=avc1]"
    "/b[height<=1080][ext=mp4]"
    "/bv*[height<=1080]"
    "/b"
)

DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fma_url_cache")


class VideoUrlError(RuntimeError):
    """Link çözülemediğinde / indirilemediğinde kullanıcıya gösterilecek hata."""


def is_url(value) -> bool:
    """Girdi http:// veya https:// ile başlıyorsa True döner."""
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


def cache_path_for(url: str, cache_dir: str = None) -> str:
    """Link için önbellekteki video dosyasının yolunu döner (aynı link -> aynı dosya)."""
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir or DEFAULT_CACHE_DIR, f"{digest}.mp4")


def _load_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoUrlError(
            "Link ile video kullanmak için 'yt-dlp' paketi gerekli ama kurulu değil. "
            "Kurmak için: pip install -r requirements.txt"
        ) from exc
    return yt_dlp


def _download_error_message(exc) -> str:
    """yt-dlp hatasını kullanıcının anlayacağı Türkçe mesaja çevirir."""
    detail = str(exc)
    lowered = detail.lower()
    if "drm" in lowered:
        reason = (
            "Bu video korumalı (DRM); indirilemez. Ücretli yayın platformlarındaki "
            "korumalı videolar desteklenmez."
        )
    elif "unsupported url" in lowered or "no video formats" in lowered:
        reason = (
            "Bu sayfada indirilebilir video bulunamadı. Videonun kendi sayfasının "
            "veya doğrudan video dosyasının (.mp4, .m3u8 ...) linkini deneyin."
        )
    else:
        reason = (
            "Video linkten indirilemedi. Linkin doğru olduğunu, videonun herkese "
            "açık olduğunu ve internet bağlantınızı kontrol edin."
        )
    return f"{reason} Ayrıntı: {detail}"


def resolve_input(input_arg: str, cache_dir: str = None) -> str:
    """
    Girdi bir link ise videoyu önbelleğe indirip yerel dosya yolunu döner;
    link değilse girdiyi olduğu gibi döner (yerel dosya davranışı değişmez).
    """
    if not is_url(input_arg):
        return input_arg

    url = input_arg.strip()
    path = cache_path_for(url, cache_dir)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print(f"[URL] Önbellekteki video kullanılıyor: {path}")
        return path

    yt_dlp = _load_yt_dlp()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    options = {
        "format": YTDLP_FORMAT,
        "outtmpl": path,
        "noplaylist": True,
        # Sayfada birden çok gömülü video varsa yalnızca ilki alınır.
        "playlist_items": "1",
        "quiet": True,
        "no_warnings": True,
    }
    print(f"[URL] Video linkten indiriliyor (en fazla 1080p): {url}")
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise VideoUrlError(_download_error_message(exc)) from exc

    if not os.path.isfile(path):
        raise VideoUrlError(f"İndirme tamamlandı ama video dosyası bulunamadı: {path}")
    print(f"[URL] Video önbelleğe kaydedildi: {path}")
    return path
