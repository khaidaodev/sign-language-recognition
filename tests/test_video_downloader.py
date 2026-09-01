"""
Tests for src/video_downloader.py. Only covers the pure logic (deciding whether a URL is
YouTube, and pulling the useful line out of yt-dlp's error output), and download_youtube's
control flow with subprocess/shutil mocked out, no real network calls or yt-dlp/ffmpeg needed to
run these.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import video_downloader  # noqa: E402
from video_downloader import (  # noqa: E402
    _ffmpeg_location,
    download_youtube,
    extract_yt_dlp_error,
    is_youtube,
)


def test_is_youtube_recognises_full_urls():
    assert is_youtube("https://www.youtube.com/watch?v=abc123")


def test_is_youtube_recognises_short_urls():
    assert is_youtube("https://youtu.be/abc123")


def test_is_youtube_false_for_other_hosts():
    assert not is_youtube("https://www.handspeak.com/word/j/jacket.mp4")


def test_extract_yt_dlp_error_returns_the_last_nonblank_line():
    stderr = b"WARNING: some noise\nERROR: [youtube] abc123: Sign in to confirm you're not a bot\n\n"
    assert extract_yt_dlp_error(stderr) == "ERROR: [youtube] abc123: Sign in to confirm you're not a bot"


def test_extract_yt_dlp_error_handles_missing_stderr():
    assert extract_yt_dlp_error(None) == "no error output"


def test_extract_yt_dlp_error_handles_blank_stderr():
    assert extract_yt_dlp_error(b"   \n  \n") == "no error output"


def test_ffmpeg_location_returns_none_when_ffmpeg_is_already_on_path(monkeypatch):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert _ffmpeg_location() is None


def test_ffmpeg_location_falls_back_to_imageio_ffmpeg_when_ffmpeg_is_missing(monkeypatch):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: None)
    fake_imageio_ffmpeg = type(
        "module", (), {"get_ffmpeg_exe": staticmethod(lambda: "/fake/path/to/ffmpeg")}
    )
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_imageio_ffmpeg)
    assert _ffmpeg_location() == "/fake/path/to/ffmpeg"


def test_ffmpeg_location_returns_none_when_neither_is_available(monkeypatch):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # simulates it not being installed
    assert _ffmpeg_location() is None


def test_download_youtube_skips_when_yt_dlp_is_not_installed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: None)
    result = download_youtube("https://youtu.be/abc123", tmp_path / "out.mp4")
    assert result is False
    assert "not installed" in capsys.readouterr().out


def test_download_youtube_returns_true_when_the_file_gets_created(tmp_path, monkeypatch):
    out_path = tmp_path / "out.mp4"
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def fake_run(cmd, **kwargs):
        out_path.write_bytes(b"fake video bytes")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(video_downloader.subprocess, "run", fake_run)
    assert download_youtube("https://youtu.be/abc123", out_path) is True


def test_download_youtube_reports_the_real_yt_dlp_error_on_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, stderr=b"ERROR: [youtube] abc123: Video unavailable\n"
        )

    monkeypatch.setattr(video_downloader.subprocess, "run", fake_run)
    result = download_youtube("https://youtu.be/abc123", tmp_path / "out.mp4")
    assert result is False
    assert "Video unavailable" in capsys.readouterr().out


def test_download_youtube_handles_a_timeout_without_crashing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video_downloader.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(video_downloader.subprocess, "run", fake_run)
    result = download_youtube("https://youtu.be/abc123", tmp_path / "out.mp4")
    assert result is False
    assert "timeout" in capsys.readouterr().out
