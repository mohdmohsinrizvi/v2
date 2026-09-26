#!/usr/bin/env python3
"""Stage 14 — run the official submission validator.

Wraps ``student_resource/utils/validate_submission.py`` so the check is one
command and always uses the project's test directory.

Usage (final, against the real test set):
    .venv/bin/python scripts/14_validate_submission.py

Usage (dry run against a dev-built stand-in directory):
    .venv/bin/python scripts/14_validate_submission.py --test-dir /tmp/... --check-ids
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = REPO.parent / "student_resource" / "utils" / "validate_submission.py"
DEFAULT_TEST_DIR = REPO.parent / "student_resource" / "dataset" / "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=str(P.SUBMISSIONS / "matching_results.tsv"))
    ap.add_argument("--candidate", default=str(P.SUBMISSIONS / "candidate_pairs.tsv"))
    ap.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    ap.add_argument("--check-ids", action="store_true")
    args = ap.parse_args()

    if not VALIDATOR.exists():
        sys.exit(f"official validator not found at {VALIDATOR}")

    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching", args.matching,
        "--candidate", args.candidate,
        "--test-dir", args.test_dir,
    ]
    if args.check_ids:
        cmd.append("--check-ids")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
