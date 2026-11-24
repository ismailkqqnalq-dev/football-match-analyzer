"""
`--input` parametresine maç linki verilebilmesi (modules/url_input.py) için
birim testleri. yt-dlp çağrıları mock'lanır; ağ erişimi gerekmez.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import url_input
from modules.url_input import VideoUrlError, cache_path_for, is_url, resolve_input


class _FakeDownloadError(Exception):
    pass


def _fake_yt_dlp(on_download):
    """yt_dlp modülünü taklit eder; download() çağrısında on_download(options, urls) çalışır."""
    calls = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls.append((self.options, list(urls)))
            on_download(self.options, urls)

    module = SimpleNamespace(
        YoutubeDL=FakeYDL,
        utils=SimpleNamespace(DownloadError=_FakeDownloadError),
    )
    return module, calls


@pytest.mark.parametrize(
    "value, expected",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("http://example.com/mac.mp4", True),
        ("  HTTPS://youtu.be/abc", True),
        ("data/input_videos/match_sample.mp4", False),
        (r"C:\videolar\mac.mp4", False),
        ("ftp://example.com/mac.mp4", False),
        ("", False),
        (None, False),
    ],
)
def test_is_url(value, expected):
    assert is_url(value) is expected


def test_local_path_is_returned_unchanged(tmp_path, monkeypatch):
    def _must_not_load():
        raise AssertionError("Yerel dosya için yt-dlp yüklenmemeli")

    monkeypatch.setattr(url_input, "_load_yt_dlp", _must_not_load)
    local = "data/input_videos/match_sample.mp4"
    assert resolve_input(local, cache_dir=str(tmp_path)) == local


def test_cache_path_is_stable_and_per_url(tmp_path):
    a1 = cache_path_for("https://youtu.be/aaa", str(tmp_path))
    a2 = cache_path_for("  https://youtu.be/aaa  ", str(tmp_path))
    b = cache_path_for("https://youtu.be/bbb", str(tmp_path))
    assert a1 == a2
    assert a1 != b
    assert os.path.dirname(a1) == str(tmp_path)
    assert a1.endswith(".mp4")


def test_default_cache_dir_is_under_temp():
    import tempfile

    assert url_input.DEFAULT_CACHE_DIR.startswith(tempfile.gettempdir())


def test_url_is_downloaded_to_cache_with_expected_options(tmp_path, monkeypatch):
    def _write_file(options, urls):
        with open(options["outtmpl"], "wb") as f:
            f.write(b"video")

    fake, calls = _fake_yt_dlp(_write_file)
    monkeypatch.setattr(url_input, "_load_yt_dlp", lambda: fake)

    url = "https://www.youtube.com/watch?v=abc"
    path = resolve_input(url, cache_dir=str(tmp_path / "cache"))

    assert path == cache_path_for(url, str(tmp_path / "cache"))
    assert os.path.isfile(path)
    assert len(calls) == 1
    options, urls = calls[0]
    assert urls == [url]
    assert options["noplaylist"] is True
    assert "height<=1080" in options["format"]
    assert "avc1" in options["format"]


def test_cached_url_is_not_downloaded_again(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc"
    cached = cache_path_for(url, str(tmp_path))
    with open(cached, "wb") as f:
        f.write(b"video")

    def _must_not_load():
        raise AssertionError("Önbellekte varken yeniden indirilmemeli")

    monkeypatch.setattr(url_input, "_load_yt_dlp", _must_not_load)
    assert resolve_input(url, cache_dir=str(tmp_path)) == cached


def test_missing_yt_dlp_gives_turkish_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # import -> ImportError
    with pytest.raises(VideoUrlError, match="yt-dlp.*kurulu değil"):
        resolve_input("https://youtu.be/abc", cache_dir=str(tmp_path))


def test_download_error_gives_turkish_message(tmp_path, monkeypatch):
    def _fail(options, urls):
        raise _FakeDownloadError("ERROR: Video unavailable")

    fake, _ = _fake_yt_dlp(_fail)
    monkeypatch.setattr(url_input, "_load_yt_dlp", lambda: fake)

    with pytest.raises(VideoUrlError, match="indirilemedi") as excinfo:
        resolve_input("https://youtu.be/yok", cache_dir=str(tmp_path))
    assert "Video unavailable" in str(excinfo.value)


def test_download_without_output_file_raises(tmp_path, monkeypatch):
    fake, _ = _fake_yt_dlp(lambda options, urls: None)
    monkeypatch.setattr(url_input, "_load_yt_dlp", lambda: fake)

    with pytest.raises(VideoUrlError, match="bulunamadı"):
        resolve_input("https://youtu.be/abc", cache_dir=str(tmp_path))


def test_main_exits_with_message_on_url_error(monkeypatch, capsys):
    import main

    def _raise(_):
        raise VideoUrlError("Video linkten indirilemedi.")

    monkeypatch.setattr(main, "resolve_input", _raise)
    monkeypatch.setattr(sys, "argv", ["main.py", "--input", "https://youtu.be/yok"])
    with pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1
    assert "[HATA] Video linkten indirilemedi." in capsys.readouterr().err


def test_main_local_missing_file_message_unchanged(tmp_path, monkeypatch, capsys):
    import main

    missing = str(tmp_path / "yok.mp4")
    monkeypatch.setattr(sys, "argv", ["main.py", "--input", missing])
    with pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1
    assert f"Giriş video dosyası bulunamadı: {missing}" in capsys.readouterr().err
