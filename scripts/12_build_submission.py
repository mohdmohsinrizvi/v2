#!/usr/bin/env python3
"""Stage 12 — build the submission files from scored candidates.

Writes, in UTF-8 tab-separated form with the exact official headers:

    output/matching_results.tsv   source1_entity_id -> matched_entity_ids
    output/candidate_pairs.tsv    source1_entity_id -> candidate_entity_ids

Every Source-1 entity of the split gets a row — an empty second column means
"no match" / "no candidates", which the scorer and validator both require.

Usage:
    .venv/bin/python scripts/12_build_submission.py --split test --threshold 0.97
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P

MATCH_HEADER = "source1_entity_id\tmatched_entity_ids"
CAND_HEADER = "source1_entity_id\tcandidate_entity_ids"


def id_maps(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = P.split_dir("processed", split)
    s1 = (
        pl.read_parquet(base / "s1_ids.parquet")
        .with_columns(pl.col("sid").cast(pl.Int32))
    )
    t = pl.read_parquet(base / "t_ids.parquet").with_columns(
        pl.col("tid").cast(pl.Int32)
    )
    return s1, t


def shard_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


FEAT_STRIDE = 128  # must match STRIDE in scripts/08_generate_features.py


def _candidate_index(
    idx: int, score_path: Path, feat_dir: Path, use_stride: bool
) -> int:
    """Which candidate shard this score file came from, via the SUCCESS chain.

    scripts/10 records the feature file it read; scripts/08 records the
    candidate shard that feature was sliced from.  Following both markers
    gives the exact candidate shard index under either naming layout
    (train/dev: feature index == candidate index; test: i*STRIDE+piece).
    Without markers fall back to the arithmetic the caller detected.
    """
    mk = P.marker(score_path)
    if mk.exists():
        feat_name = P.read_json(mk).get("src")
        if feat_name:
            fm = P.marker(feat_dir / Path(str(feat_name)).name)
            if fm.exists():
                cand_name = P.read_json(fm).get("src")
                if cand_name:
                    return shard_index(Path(str(cand_name)))
    return idx // FEAT_STRIDE if use_stride else idx


def group_score_shards(
    scores: dict[int, Path],
    feat_dir: Path,
) -> dict[int, list[Path]]:
    """Regroup piece-level score files under the candidate shard they cover.

    scripts/08 --mode candidates slices each candidate shard into --chunk
    pieces and names the outputs i*STRIDE+piece; scripts/10 scores each piece
    into its own file.  A piece is a block of *rows*, not of *sids* (the
    candidate rows of a shard are ordered by retrieval method, so one piece
    already spans a large slice of the sid range), so several pieces of the
    same candidate shard carry the same sid.  Emitting one group_by per piece
    wrote 48.8M lines instead of one line per Source-1 row.  Candidate shards
    are sid-disjoint and ascending, so regrouping before the group_by restores
    exactly one line per Source-1 row.
    """
    use_stride = bool(scores) and max(scores) >= FEAT_STRIDE
    out: dict[int, list[Path]] = {}
    for idx, path in scores.items():
        cand = _candidate_index(idx, path, feat_dir, use_stride)
        out.setdefault(cand, []).append(path)
    return out


def write_submission(
    groups: dict[int, list[Path]],
    s1: pl.DataFrame,
    t: pl.DataFrame,
    threshold: float | None,
    path: Path,
    label: str,
) -> tuple[int, int, int]:
    """Scored/candidate shards -> one TSV line per Source-1 entity, in sid order.

    Streams: one candidate shard (all its score pieces concatenated) is read,
    aggregated and written before the next is opened. The train candidate set
    is ~214M ids, which cannot be held as one frame. Candidate shards are
    disjoint and ascending in ``sid``, so lines can be appended directly;
    Source-1 rows a shard never touches are written empty.
    """
    n_s1 = s1.height
    eids = s1["entity_id"].to_list()
    header = CAND_HEADER if path.name.startswith("candidate") else MATCH_HEADER
    P.ensure(path.parent)
    ptr = 0
    n_rows = 0
    n_with = 0
    n_ids = 0
    n_files = sum(len(v) for v in groups.values())
    seen = 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        w = fh.write
        for c in sorted(groups):
            cols = ["sid", "tid"] + (["score"] if threshold is not None else [])
            frames = []
            for f in sorted(groups[c]):
                df = pl.read_parquet(f, columns=cols)
                if threshold is not None:
                    df = df.filter(pl.col("score") >= threshold)
                if df.height:
                    frames.append(df)
            seen += len(groups[c])
            if frames:
                df = frames[0] if len(frames) == 1 else pl.concat(frames)
                del frames
                g = (
                    df.join(t, on="tid", how="inner")
                    .group_by("sid")
                    .agg(pl.col("entity_id").unique().sort().alias("ids"))
                    .sort("sid")
                )
                if g.height:
                    sids = g["sid"].to_list()
                    idlists = g["ids"].to_list()
                    for sid, vals in zip(sids, idlists):
                        if sid >= n_s1:
                            sys.exit(f"sid {sid} out of range (s1={n_s1})")
                        if sid < ptr:
                            sys.exit(
                                f"{label}: candidate group {c} repeats sid {sid} "
                                f"already written at ptr={ptr}; groups overlap"
                            )
                        while ptr < sid:
                            w(f"{eids[ptr]}\t\n")
                            ptr += 1
                            n_rows += 1
                        w(f"{eids[sid]}\t{','.join(vals)}\n")
                        ptr = sid + 1
                        n_rows += 1
                        if vals:
                            n_with += 1
                        n_ids += len(vals)
                    del g, sids, idlists
                del df
            if seen % 20 == 0 or seen == n_files:
                print(f"    {label}: {seen}/{n_files} shards", flush=True)

        while ptr < n_s1:
            w(f"{eids[ptr]}\t\n")
            ptr += 1
            n_rows += 1

    if n_rows != n_s1:
        sys.exit(f"{label}: wrote {n_rows} rows, expected {n_s1}")
    return n_rows, n_with, n_ids


BUCKET = "amazon-ml-challenge-2026-357112746764"


def merge_helper_predictions(timeout_s: int = 10800) -> None:
    """Two-instance run: pull the helper machine's half of the scores.

    Only runs when ``artifacts/helper.expected`` exists, so a normal
    single-instance test run never blocks here.
    """
    import subprocess
    marker = f"s3://{BUCKET}/run/status/H_PREDS_DONE.txt"
    remote = f"s3://{BUCKET}/run/artifacts/predictions/test/"
    local = P.split_dir("predictions", "test")
    deadline = time.time() + timeout_s
    while True:
        r = subprocess.run(["aws", "s3", "ls", marker],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        if time.time() > deadline:
            print("[12] WARNING: helper marker never appeared; "
                  "building from local scores only", flush=True)
            return
        print("[12] waiting for helper predictions...", flush=True)
        time.sleep(60)
    print("[12] helper finished; merging its predictions", flush=True)
    subprocess.run(["aws", "s3", "sync", remote, str(local),
                    "--only-show-errors"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: best from reports/threshold_{split}.json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.threshold is None:
        thr_path = P.REPORTS / f"threshold_{args.split}.json"
        if not thr_path.exists() and args.split == "test":
            # The cutoff must never be fit on the test split, so the value
            # chosen on train is the one that governs the test submission.
            thr_path = P.REPORTS / "threshold_train.json"
        if not thr_path.exists():
            sys.exit(f"no threshold file {thr_path}; run scripts/11 or pass --threshold")
        args.threshold = float(P.read_json(thr_path)["best_threshold"])
    out_dir = Path(args.out_dir) if args.out_dir else P.SUBMISSIONS
    print(f"[12] split={args.split} threshold={args.threshold}", flush=True)

    if args.split == "test" and (P.ARTIFACTS / "helper.expected").exists():
        merge_helper_predictions()

    s1, t = id_maps(args.split)
    score_files = P.shard_files(P.split_dir("predictions", args.split))
    cand_files = P.shard_files(P.split_dir("candidates", args.split))
    if not score_files or not cand_files:
        sys.exit("need both scores (scripts/10) and candidates (scripts/05)")

    # scripts/08 writes candidate features as shard i*STRIDE+piece while
    # scripts/05 writes contiguous candidate shards 0..N-1, so the two index
    # spaces do NOT line up: intersecting them kept only the handful of score
    # shards whose index happened to be < N and silently dropped the rest.
    # The two outputs are independent -- give each its full set -- but score
    # pieces must first be regrouped under their candidate shard, because a
    # piece is a slice of rows and several pieces share a sid.
    sm = {shard_index(f): f for f in score_files}
    sc = {shard_index(f): f for f in cand_files}
    if not sm or not sc:
        sys.exit("no score/candidate shards")
    feat_dir = P.split_dir("features", args.split) / "candidates"
    # scripts/10 partitions its input by i % num_workers, so a run that
    # launched the wrong worker set scores a clean half and leaves the rest
    # with no scores at all -- those Source-1 rows would come out empty.
    # This machine can only check its OWN half: the helper uploaded the other
    # half and validated it before it published its completion marker. Compare
    # scored pieces against local feature pieces, per candidate shard.
    local_cov: dict[int, int] = {}
    for f in P.shard_files(feat_dir):
        mk = P.marker(f)
        if not mk.exists():
            continue
        src_name = P.read_json(mk).get("src")
        if not src_name:
            continue
        c = shard_index(Path(str(src_name)))
        local_cov[c] = local_cov.get(c, 0) + 1
    g_sm_early = group_score_shards(sm, feat_dir)
    short = {
        c: (local_cov[c], len(g_sm_early.get(c, [])))
        for c in local_cov
        if len(g_sm_early.get(c, [])) < local_cov[c]
    }
    if short:
        sys.exit(
            f"candidate shards with fewer scores than local feature pieces "
            f"(expected, scored): {short} -- rerun scripts/10"
        )
    g_sm = g_sm_early
    g_sc = {c: [f] for c, f in sc.items()}
    if set(g_sm) != set(g_sc):
        sys.exit(
            f"scored candidate shards {sorted(g_sm)} != candidate shards "
            f"{sorted(g_sc)}"
        )
    print(
        f"[12] {len(sm)} score files -> {len(g_sm)} candidate groups, "
        f"{len(sc)} candidate shards",
        flush=True,
    )

    print("[12] writing matches...", flush=True)
    s1_rows, n_match_rows, n_ids = write_submission(
        g_sm, s1, t, args.threshold,
        out_dir / "matching_results.tsv", "matches")
    print("[12] writing candidates...", flush=True)
    _, n_cand_rows, n_cand = write_submission(
        g_sc, s1, t, None,
        out_dir / "candidate_pairs.tsv", "candidates")
    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "s1_rows": int(s1_rows),
        "s1_with_matches": n_match_rows,
        "s1_with_candidates": n_cand_rows,
        "match_ids": n_ids,
        "candidate_ids": n_cand,
        "avg_matches_per_s1": round(n_ids / max(s1_rows, 1), 3),
        "avg_candidates_per_s1": round(n_cand / max(s1_rows, 1), 3),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    P.write_json(out_dir / "submission_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
