"""Guards added after the train recall-gate failure (0.8238 < 0.97).

Two regressions they catch:

* r_tfidf_rank_norm is normalised by tfidf_k, so features built with one k can
  be scored as if built with another -- silently degrades every score;
* scripts/08 --val-only used to hard-code the fold at 0.10, which disagreed with
  scripts/07/11 whenever another fold was requested, so only ~52% of the val
  fold's ground truth was ever scored.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pipeline import paths as P


def _load(script: str):
    spec = importlib.util.spec_from_file_location(
        f"_t_{script.strip('.py')}", PROJECT / "scripts" / script
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shard(dirpath: pathlib.Path, rows: int, tfidf_k: int) -> pathlib.Path:
    f = dirpath / "shard_00000.parquet"
    f.write_bytes(b"not-a-real-parquet")
    P.mark_done(f, {"rows": rows, "tfidf_k": tfidf_k})
    return f


def test_check_feature_k_accepts_matching_k(tmp_path: pathlib.Path) -> None:
    _shard(tmp_path, rows=10, tfidf_k=100)
    assert P.check_feature_k(tmp_path, 100, "test") == 100


def test_check_feature_k_rejects_mismatched_k(tmp_path: pathlib.Path) -> None:
    _shard(tmp_path, rows=10, tfidf_k=50)
    with pytest.raises(SystemExit, match="tfidf-k 50"):
        P.check_feature_k(tmp_path, 100, "test")


def test_check_feature_k_tolerates_missing_marker(tmp_path: pathlib.Path) -> None:
    (tmp_path / "shard_00000.parquet").write_bytes(b"")
    assert P.check_feature_k(tmp_path, 100, "test") == 100


def test_check_feature_k_empty_dir_is_not_an_error(tmp_path: pathlib.Path) -> None:
    assert P.check_feature_k(tmp_path, 100, "test") == 100


def test_prediction_shard_is_reused_only_for_the_same_row_count(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load("10_score_candidates.py")
    feat = _shard(tmp_path, rows=100, tfidf_k=100)
    pred = tmp_path / "out" / "shard_00000.parquet"
    pred.parent.mkdir()
    P.mark_done(pred, {"rows": 100, "src": feat.name})
    assert mod._rows_match(pred, feat) is True

    P.mark_done(feat, {"rows": 250, "tfidf_k": 100})
    assert mod._rows_match(pred, feat) is False


def test_dev_target_rows_match_the_df_reference() -> None:
    """04 scales the df cutoffs by target_rows / 272_474, and dev is exactly
    that reference, so scaling must be a no-op there -- otherwise every
    previously recorded dev number stops being comparable."""
    assert _load("04_build_indices.py").target_rows("dev") == 272_474


def test_08_exposes_val_frac() -> None:
    out = subprocess.run(
        [sys.executable, "scripts/08_generate_features.py", "--help"],
        cwd=PROJECT, capture_output=True, text=True, check=True,
    ).stdout
    assert "--val-frac" in out
