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


_TEMP_SUFFIXES = (".part", ".ytdl", ".tmp")


def _url_digest(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def cache_path_for(url: str, cache_dir: str = None, ext: str = "mp4") -> str:
    """Link için önbellekteki video dosyasının yolunu döner (aynı link -> aynı dosya)."""
    digest = _url_digest(url)
    clean_ext = str(ext or "mp4").lstrip(".")
    return os.path.join(cache_dir or DEFAULT_CACHE_DIR, f"{digest}.{clean_ext}")


def _find_cached_file(url: str, cache_dir: str = None):
    """Aynı link için önbellekteki (mp4/webm/mkv vb.) hazır video dosyasını bulur."""
    base_dir = cache_dir or DEFAULT_CACHE_DIR
    mp4_path = cache_path_for(url, base_dir, ext="mp4")
    if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0:
        return mp4_path
    if not os.path.isdir(base_dir):
        return None
    prefix = f"{_url_digest(url)}."
    for name in sorted(os.listdir(base_dir)):
        if not name.startswith(prefix):
            continue
        if any(name.lower().endswith(suf) for suf in _TEMP_SUFFIXES):
            continue
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


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
    base_dir = cache_dir or DEFAULT_CACHE_DIR
    cached = _find_cached_file(url, base_dir)
    if cached is not None:
        print(f"[URL] Önbellekteki video kullanılıyor: {cached}")
        return cached

    yt_dlp = _load_yt_dlp()
    os.makedirs(base_dir, exist_ok=True)
    outtmpl = os.path.join(base_dir, f"{_url_digest(url)}.%(ext)s")
    options = {
        "format": YTDLP_FORMAT,
        "outtmpl": outtmpl,
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

    downloaded = _find_cached_file(url, base_dir)
    if downloaded is None:
        raise VideoUrlError(f"İndirme tamamlandı ama video dosyası bulunamadı: {outtmpl}")
    print(f"[URL] Video önbelleğe kaydedildi: {downloaded}")
    return downloaded
