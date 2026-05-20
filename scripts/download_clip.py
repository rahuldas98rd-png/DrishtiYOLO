"""
download_clip.py — yt-dlp wrapper for sourcing CC-licensed Indian traffic clips.

Usage:
    python scripts/download_clip.py --url <youtube_url> --out assets/clips/
    python scripts/download_clip.py --search "Indian traffic Siliguri" --out assets/clips/

    # Grab only the first 60 s of any video (ignores total length filter):
    python scripts/download_clip.py --url <youtube_url> --trim 60
    python scripts/download_clip.py --search "India dashcam traffic" --n 1 --trim 60

Only downloads videos with Creative Commons licences when using --search.
"""

import argparse
import subprocess
import sys
from pathlib import Path


CC_FILTER = "creativecommons"   # yt-dlp licence filter keyword


def _base_cmd(out_dir: Path, trim: int | None) -> list[str]:
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--output", str(out_dir / "%(title)s.%(ext)s"),
        "--no-playlist",
    ]
    if trim:
        # Download only the first `trim` seconds — works on any length video
        cmd += ["--download-sections", f"*0-{trim}"]
    return cmd


def download(url: str, out_dir: Path, max_duration: int = 300, trim: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(out_dir, trim)
    if not trim:
        cmd += ["--match-filter", f"duration <= {max_duration}"]
    cmd.append(url)
    print(f"Downloading: {url}" + (f" (first {trim}s)" if trim else ""))
    subprocess.run(cmd, check=True)


def search_and_download(query: str, out_dir: Path, n: int = 3, trim: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    search_url = f"ytsearch{n}:{query} {CC_FILTER} license"
    cmd = _base_cmd(out_dir, trim)
    if not trim:
        cmd += ["--match-filter", "duration <= 300"]
    cmd.append(search_url)
    print(f"Searching YouTube for: {query!r} (CC only, up to {n} results)" +
          (f" — trimming to first {trim}s" if trim else ""))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download Indian traffic clips via yt-dlp")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",    help="Direct YouTube URL")
    group.add_argument("--search", help="Search query (CC-licensed results only)")
    parser.add_argument("--out",          default="assets/clips", help="Output directory")
    parser.add_argument("--n",            type=int, default=3,    help="Number of search results to download")
    parser.add_argument("--max-duration", type=int, default=300,  help="Max clip length in seconds (ignored when --trim is set)")
    parser.add_argument("--trim",         type=int, default=None, metavar="SECONDS",
                        help="Download only the first N seconds (bypasses the duration filter)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    try:
        if args.url:
            download(args.url, out_dir, args.max_duration, args.trim)
        else:
            search_and_download(args.search, out_dir, args.n, args.trim)
    except subprocess.CalledProcessError as e:
        print(f"yt-dlp failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Clips saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
