#!/usr/bin/env python3
"""
Download ML weights into ./models from the public Drive folder linked in README.md.

Requires: pip install gdown
Usage:
  python tools/download_models.py
  python tools/download_models.py --output /path/to/models
"""

from __future__ import annotations

import argparse
import os
import sys

# Folder: "Models file" in README (AhmedHamadaIT/aged_gender)
DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1oAROlqkBo8C3rzTe4hAcS7abaIKC_Ugq"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download models from Google Drive into models/")
    parser.add_argument(
        "--url",
        default=os.environ.get("MODELS_DRIVE_FOLDER_URL", DEFAULT_DRIVE_FOLDER),
        help="Google Drive folder URL or ID",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
        help="Destination directory (created if missing)",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("Install gdown first: pip install gdown", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading folder into: {args.output}")
    gdown.download_folder(url=args.url, output=args.output, quiet=False, remaining_ok=True)
    print("Done. Verify files (examples): osnet_x1_0.pt, image_encoder.onnx, text_encoder.onnx, yolov8n.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
