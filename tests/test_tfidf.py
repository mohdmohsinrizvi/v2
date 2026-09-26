"""TF-IDF retriever behaviour + save/load parity.

Exercises the country-partitioned fit path that stage 04 runs on the full
corpus (rewritten to bound peak RSS), so a memory optimisation that silently
changes retrieval is caught here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.tfidf import CountryRetriever, TfidfConfig, TfidfIndex

# The production config (min_df=3) prunes terms inside each country partition;
# every real partition here has >=1.4M rows, but this 6-row fixture does not.
CFG = TfidfConfig(min_df=1, k=3)


def corpus() -> pl.DataFrame:
    rows = [
        (0, "DE", "acme gmbh", "hauptstrasse 1 berlin"),
        (1, "DE", "acme gmbh", "hauptstrasse 2 berlin"),
        (2, "DE", "beta llc", "unter den angi 5"),
        (3, "US", "acme inc", "main street 1 boston"),
        (4, "US", "gamma corp", "park avenue 99 new york"),
        (5, "FR", "societe generale", "rue de la paix 12"),
    ]
    return pl.DataFrame(
        rows, schema=["tid", "country", "name_tok", "addr_tok"], orient="row"
    )


def test_fit_indexes_every_country() -> None:
    r = CountryRetriever(TfidfConfig(min_df=1, k=2))
    info = r.fit(corpus())

    assert set(info) == {"DE", "US", "FR"}
    assert set(r.indexes) == {"DE", "US", "FR"}
    for key in ("DE", "US", "FR"):
        assert r.indexes[key].n_target == info[key]["n_target"]


def test_query_stays_within_country() -> None:
    r = CountryRetriever(CFG)
    r.fit(corpus())

    q = pl.DataFrame(
        [(100, "DE", "acme gmbh", "hauptstrasse 3 berlin")],
        schema=["sid", "country", "name_tok", "addr_tok"],
        orient="row",
    )
    out = r.query(q)

    assert out.height >= 1
    got = set(out["tid"].to_list())
    assert got <= {0, 1, 2}, "cross-country hits leaked into the K slots"


def test_query_ignores_unknown_country() -> None:
    r = CountryRetriever(CFG)
    r.fit(corpus())
    q = pl.DataFrame(
        [(7, "XX", "acme", "somewhere")],
        schema=["sid", "country", "name_tok", "addr_tok"],
        orient="row",
    )
    assert r.query(q).height == 0


def test_save_load_roundtrip_matches(tmp_path: Path) -> None:
    r = CountryRetriever(CFG)
    r.fit(corpus())
    q = pl.DataFrame(
        [(1, "DE", "acme gmbh", "hauptstrasse 1 berlin"), (2, "US", "gamma corp", "park avenue 99")],
        schema=["sid", "country", "name_tok", "addr_tok"],
        orient="row",
    )
    cols = ["sid", "tid", "tfidf_rank", "tfidf_score"]
    before = r.query(q).select(cols).sort(["sid", "tid"])

    r.save(tmp_path / "tfidf")
    reloaded = CountryRetriever.load(tmp_path / "tfidf", CFG)
    after = reloaded.query(q).select(cols).sort(["sid", "tid"])

    assert before.equals(after)
    assert set(reloaded.indexes) == set(r.indexes)


def test_index_fit_releases_nothing_that_query_needs() -> None:
    idx = TfidfIndex(TfidfConfig(min_df=1, k=2))
    info = idx.fit(["a b c", "b c d", "c d e", "z y x"])

    assert info["n_target"] == 4
    assert idx.Xt is not None and idx.Xt_T is not None
    assert idx.Xt.shape == idx.Xt_T.T.shape
    assert len(idx.vocab) == info["kept_vocab"]
