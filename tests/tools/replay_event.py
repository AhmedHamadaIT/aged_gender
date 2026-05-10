#!/usr/bin/env python3
"""Pretty-print a JSON task/detection event (for triage). Usage: replay_event.py [file|-]"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Path to JSON file, or '-' for stdin",
    )
    args = p.parse_args()
    if args.path == "-":
        raw = sys.stdin.read()
    else:
        raw = open(args.path, encoding="utf-8").read()
    obj = json.loads(raw)
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
