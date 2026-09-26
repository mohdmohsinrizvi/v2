"""Canonical artifact layout + resumable shard manifests."""
from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
SUBMISSIONS = PROJECT_ROOT / "submissions"
LOGS = PROJECT_ROOT / "logs"
MODELS = ARTIFACTS / "model"


def split_dir(kind: str, split: str) -> Path:
    return ARTIFACTS / kind / split


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def shard_path(base: Path, i: int) -> Path:
    return base / f"shard_{i:05d}.parquet"


def marker(path: Path) -> Path:
    return Path(str(path) + ".SUCCESS")


def is_done(path: Path) -> bool:
    return marker(path).exists()


def mark_done(path: Path, info: dict | None = None) -> None:
    payload = dict(info or {})
    payload.setdefault("finished_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    marker(path).write_text(json.dumps(payload, indent=2))


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def manifest(base: Path) -> dict:
    p = base / "manifest.json"
    if p.exists():
        return read_json(p)
    return {"shards": []}


def save_manifest(base: Path, obj: dict) -> None:
    write_json(base / "manifest.json", obj)


def check_feature_k(base: Path, expected: int, what: str) -> int:
    """r_tfidf_rank_norm is normalised by tfidf_k, so a features shard built
    with a different k than the model (or the caller) silently degrades every
    score. Read the k recorded in the first shard marker and refuse to go on.
    """
    files = shard_files(base)
    if not files:
        return expected
    mk = marker(files[0])
    if not mk.exists():
        return expected
    got = read_json(mk).get("tfidf_k")
    if got is not None and int(got) != int(expected):
        raise SystemExit(
            f"{what}: features in {base} were built with --tfidf-k {got}, "
            f"expected {expected}; delete them and rerun scripts/08"
        )
    return expected


def shard_files(base: Path) -> list[Path]:
    return sorted(base.glob("shard_*.parquet"))
