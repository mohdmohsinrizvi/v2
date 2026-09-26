"""Regression for the test-split row blow-up in stage 12.

Stage 12 must emit exactly one line per Source-1 row. Score files are written
one per *piece* of a candidate shard (scripts/08 slices a shard into --chunk
pieces), and a piece is a slice of rows, not of sids -- the candidate rows of a
shard are ordered by retrieval method, so one piece already spans much of the
sid range and several pieces of the same candidate shard contain the same sid.
Aggregating each piece on its own emitted one line per (piece, sid): the test
run wrote 48,756,072 lines for 1,732,544 Source-1 rows and aborted. Candidate
shards are sid-disjoint, so regrouping the pieces under their candidate shard
before the group_by restores exactly one line per Source-1 row.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import polars as pl
import pytest

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pipeline import paths as P


def _load():
    spec = importlib.util.spec_from_file_location(
        "_t_12", PROJECT / "scripts" / "12_build_submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feature(feat_dir: pathlib.Path, name: str, cand: str) -> None:
    """scripts/08 records which candidate shard a feature piece came from."""
    P.mark_done(feat_dir / f"{name}.parquet", {"src": f"{cand}.parquet"})


def _score(
    score_dir: pathlib.Path, idx: int, feat: str, sids: list[int], tids: list[int]
) -> pathlib.Path:
    f = score_dir / f"shard_{idx:05d}.parquet"
    pl.DataFrame(
        {
            "sid": sids,
            "tid": tids,
            "score": [0.9] * len(sids),
        }
    ).write_parquet(f)
    P.mark_done(f, {"src": f"{feat}.parquet"})
    return f


@pytest.fixture()
def split_dirs(tmp_path: pathlib.Path):
    feat = tmp_path / "features" / "test" / "candidates"
    feat.mkdir(parents=True)
    scores = tmp_path / "predictions" / "test"
    scores.mkdir(parents=True)
    # candidate shard 0 -> feature pieces 0 and 1 (they SHARE sids 1 and 2)
    _feature(feat, "shard_00000", "shard_00000")
    _feature(feat, "shard_00001", "shard_00000")
    # candidate shard 1 -> feature piece 130 (i*STRIDE+piece with i=1)
    _feature(feat, "shard_00130", "shard_00001")
    f0 = _score(scores, 0, "shard_00000", [0, 1, 2], [10, 11, 12])
    f1 = _score(scores, 1, "shard_00001", [1, 2, 3], [13, 14, 15])
    f2 = _score(scores, 130, "shard_00130", [4, 5], [16, 17])
    return feat, {0: f0, 1: f1, 130: f2}


def _frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.DataFrame(
        {
            "sid": list(range(6)),
            "entity_id": [f"S1-{i}" for i in range(6)],
        }
    )
    t = pl.DataFrame(
        {
            "tid": list(range(10, 18)),
            "entity_id": [f"T{i}" for i in range(10, 18)],
        }
    )
    return s1, t


def test_score_pieces_regroup_to_their_candidate_shard(split_dirs) -> None:
    mod = _load()
    feat, scores = split_dirs
    groups = mod.group_score_shards(scores, feat)
    assert sorted(groups) == [0, 1]
    assert sorted(p.name for p in groups[0]) == [
        "shard_00000.parquet",
        "shard_00001.parquet",
    ]
    assert [p.name for p in groups[1]] == ["shard_00130.parquet"]


def test_regrouped_submission_has_one_line_per_source1_row(
    split_dirs, tmp_path: pathlib.Path
) -> None:
    mod = _load()
    feat, scores = split_dirs
    s1, t = _frames()
    out = tmp_path / "matching_results.tsv"
    n_rows, n_with, n_ids = mod.write_submission(
        mod.group_score_shards(scores, feat), s1, t, 0.5, out, "matches"
    )
    assert n_rows == s1.height
    assert n_with == 6
    assert n_ids == 8  # sids 1 and 2 contribute two ids each, after the union

    lines = out.read_text().splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == s1.height + 1
    # sid 1 and 2 arrive from two different pieces: ids must be unioned,
    # never written twice.
    assert lines[2].split("\t")[1] == "T11,T13"  # sid 1
    assert lines[3].split("\t")[1] == "T12,T14"  # sid 2
    assert [ln.split("\t")[0] for ln in lines[1:]] == [
        f"S1-{i}" for i in range(6)
    ]


def test_ungrouped_pieces_are_rejected_loudly(
    split_dirs, tmp_path: pathlib.Path
) -> None:
    """The pre-fix behaviour: each piece aggregated alone repeats a sid."""
    mod = _load()
    _, scores = split_dirs
    s1, t = _frames()
    ungrouped = {i: [f] for i, f in scores.items()}
    with pytest.raises(SystemExit, match="repeats sid"):
        mod.write_submission(
            ungrouped, s1, t, 0.5, tmp_path / "m.tsv", "matches"
        )


def test_falls_back_to_stride_arithmetic_without_markers(
    tmp_path: pathlib.Path,
) -> None:
    """A marker-less score file is still mapped by the detected layout."""
    mod = _load()
    feat = tmp_path / "features" / "test" / "candidates"
    feat.mkdir(parents=True)
    scores = tmp_path / "predictions" / "test"
    scores.mkdir(parents=True)
    f = pl.DataFrame({"sid": [0], "tid": [10], "score": [0.9]})
    stride_layout = scores / "shard_00258.parquet"
    f.write_parquet(stride_layout)
    assert mod._candidate_index(258, stride_layout, feat, True) == 2
    direct_layout = scores / "shard_00005.parquet"
    f.write_parquet(direct_layout)
    assert mod._candidate_index(5, direct_layout, feat, False) == 5
