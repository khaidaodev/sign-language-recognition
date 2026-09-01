"""
Downloads the actual video clips for whichever subset wlasl_metadata.py picked out, and trims
each one down to the frame range the dataset says the sign actually happens in (frame_start /
frame_end, some clips have extra footage either side of the sign itself, frame_end of -1 means
"goes to the end of the clip").

This dataset is old and the videos live on a scattered mix of small ASL dictionary sites, a lot
of those links are dead by now (sites shut down, moved, started blocking hotlinking), and the
YouTube ones need yt-dlp installed separately. So this is written to expect a lot of failures:
it downloads what it can, skips and logs anything that fails, and keeps going instead of stopping
on the first dead link. Whatever gets downloaded successfully is enough to prove the pipeline
works end to end, doesn't need every single clip to succeed.

Run it with:
    python src/video_downloader.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlasl_metadata import SUBSET_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw" / "videos"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_direct(url: str, out_path: Path) -> bool:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"  failed ({e.__class__.__name__}): {url}")
        return False


def _ffmpeg_location() -> str | None:
    """yt-dlp needs ffmpeg to merge separate video+audio streams into one file (see
    download_youtube below). If it's not on PATH, fall back to the standalone binary the
    imageio-ffmpeg pip package bundles, no admin install or package manager needed, just
    `pip3 install imageio-ffmpeg`. Returns None if neither is available (yt-dlp will then just
    fail on any video that actually needs merging, same as before)."""
    if shutil.which("ffmpeg") is not None:
        return None  # yt-dlp finds it on PATH itself, nothing extra to pass
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def download_youtube(url: str, out_path: Path) -> bool:
    if shutil.which("yt-dlp") is None:
        print(f"  skipped, yt-dlp not installed: {url}")
        return False

    command = [
        "yt-dlp",
        # plenty of YouTube videos these days don't have a single combined video+audio mp4
        # format (just requesting "-f mp4" fails for those), so this asks for the best mp4
        # video + best m4a audio and has yt-dlp merge them (needs ffmpeg), falling back to
        # whatever's best if that specific combo isn't available
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
    ]
    ffmpeg_location = _ffmpeg_location()
    if ffmpeg_location:
        command += ["--ffmpeg-location", ffmpeg_location]
    command += ["-o", str(out_path), url]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        return out_path.exists()
    except subprocess.CalledProcessError as e:
        # yt-dlp's actual complaint (age-restricted, blocked, video removed, needs sign-in, etc.)
        # is on stderr, "exit status 1" on its own doesn't say whether this is a genuinely dead
        # video or something worth digging into
        detail = extract_yt_dlp_error(e.stderr)
        print(f"  failed (yt-dlp): {url} ({detail})")
        return False
    except subprocess.TimeoutExpired:
        print(f"  failed (yt-dlp timeout): {url}")
        return False


def extract_yt_dlp_error(stderr: bytes | None) -> str:
    """Pulls the last non-empty line out of yt-dlp's stderr, that's reliably where its actual
    error message ends up (things like "ERROR: [youtube] ...: Sign in to confirm you're not a
    bot"), everything above it is usually just progress/warning noise."""
    if not stderr:
        return "no error output"
    lines = [line for line in stderr.decode(errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else "no error output"


def trim_clip(path: Path, frame_start: int, frame_end: int) -> None:
    if frame_start <= 1 and frame_end == -1:
        return

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = path.with_suffix(".trim.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < frame_start:
            continue
        if frame_end != -1 and frame_idx > frame_end:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    tmp_path.replace(path)


def download_subset(subset_path: Path = SUBSET_PATH):
    glosses = json.loads(subset_path.read_text())
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    ok_count, fail_count = 0, 0
    for gloss_entry in glosses:
        gloss = gloss_entry["gloss"]
        gloss_dir = VIDEO_DIR / gloss
        gloss_dir.mkdir(exist_ok=True)

        for inst in gloss_entry["instances"]:
            out_path = gloss_dir / f"{inst['video_id']}.mp4"
            if out_path.exists():
                ok_count += 1
                continue

            print(f"{gloss}/{inst['video_id']}: {inst['url']}")
            success = (
                download_youtube(inst["url"], out_path)
                if is_youtube(inst["url"])
                else download_direct(inst["url"], out_path)
            )

            if success:
                try:
                    trim_clip(out_path, inst["frame_start"], inst["frame_end"])
                    ok_count += 1
                except Exception as e:
                    print(f"  trim failed: {e}")
                    fail_count += 1
            else:
                fail_count += 1

            time.sleep(0.3)

    print(f"\ndone: {ok_count} downloaded, {fail_count} failed/skipped")
    return ok_count, fail_count


if __name__ == "__main__":
    download_subset()
