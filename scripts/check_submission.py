#!/usr/bin/env python3
"""CLI to validate participant submissions (workspace or zip)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_rescue.submission_check import validate_submission  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a Swarm Rescue submission is valid "
            "(project workspace before zipping, or teamNNN_evalstep.zip)."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT),
        help="Project root or path to teamNNN_evalstep.zip (default: project root)",
    )
    args = parser.parse_args(argv)

    target = Path(args.path).resolve()
    result = validate_submission(target)

    for message in result.errors:
        print(f"ERROR: {message}", file=sys.stderr)
    for message in result.warnings:
        print(f"WARNING: {message}")

    if result.ok:
        print("Submission check passed.")
        return 0

    print("Submission check failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
