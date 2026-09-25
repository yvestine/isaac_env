#!/usr/bin/env python3
"""Transcode one generated video to H.264/yuv420p in place."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    temporary = args.video.with_name(f"{args.video.stem}.h264.tmp.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(args.video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(args.video)
    print(f"[H264] {args.video.resolve()}")


if __name__ == "__main__":
    main()
