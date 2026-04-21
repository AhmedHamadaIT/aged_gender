#!/usr/bin/env python3
"""
Download ML weights into ./models from the Google Drive folder linked in README.md.

Requires: pip install gdown

Drive must allow anonymous listing (401 = folder is private or link-restricted).
Then either: General access → “Anyone with the link” (Viewer), or use browser
cookies with gdown (see https://github.com/wkentaro/gdown#authentication).

Usage:
  python tools/download_models.py
  python tools/download_models.py --output /path/to/models
  python tools/download_models.py --no-cookies   # if gdown supports use_cookies
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

try:
    from gdown.exceptions import DownloadError as GdownDownloadError
except ImportError:
    GdownDownloadError = None  # type: ignore[misc,assignment]

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
    parser.add_argument(
        "--no-cookies",
        action="store_true",
        help="Pass use_cookies=False to gdown (if supported); try if default session gets 401.",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("Install gdown first: pip install gdown", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading folder into: {args.output}")
    sig = inspect.signature(gdown.download_folder)
    kwargs: dict = {"url": args.url, "output": args.output, "quiet": False}
    if "remaining_ok" in sig.parameters:
        kwargs["remaining_ok"] = True
    if args.no_cookies and "use_cookies" in sig.parameters:
        kwargs["use_cookies"] = False

    _errors: tuple[type[BaseException], ...] = (RuntimeError,)
    if GdownDownloadError is not None:
        _errors = (RuntimeError, GdownDownloadError)

    try:
        gdown.download_folder(**kwargs)
    except _errors as exc:
        msg = str(exc)
        hint401 = ""
        if "401" in msg or "Unauthorized" in msg:
            hint401 = (
                "\n(HTTP 401: Google rejected the request. The folder is not publicly listable.)\n"
                "  • In Drive: Share → General access → “Anyone with the link” (Viewer).\n"
                "  • Or use gdown with cookies from a logged-in browser session (gdown docs).\n"
                "  • Or copy osnet_x1_0.pt, image_encoder.onnx, text_encoder.onnx into models/ by hand.\n"
            )
        print(
            "gdown could not download the Drive folder.\n"
            "Common causes: private/restricted link, quota, or Google HTML changes.\n"
            f"{hint401}"
            "Original error:",
            exc,
            file=sys.stderr,
            sep="\n",
        )
        return 1
    print("Done. Verify files (examples): osnet_x1_0.pt, image_encoder.onnx, text_encoder.onnx, yolov8n.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
