"""Word TF-IDF top-K retrieval (sparse, cosine, country-blocked).

Why not ``sklearn.NearestNeighbors``: its brute-force path materializes
``chunk x n_target`` floats, which explodes at ~10M targets. Here both matrices
are L2-normalized sparse matrices, so ``Q @ T.T`` *is* the cosine similarity and
stays sparse (only co-occurring documents materialize); top-K is then read from
each row's non-zeros.

Two design choices that dominate measured recall (see ``reports/data_audit.md``
dev-sample experiment):

* **Country blocking** — every ground-truth link is same-country, so each country
  gets its own index and its own K slots. Without this, foreign rows steal the
  K slots and pair recall drops from ~0.98 to ~0.85.
* **Name + address text** — indexing ``name_tok + addr_tok`` together lifts pair
  recall well above name-only indexing.

Vocabulary is restricted to ``min_df <= df <= max_df`` on the target side so the
sparse product stays cheap (bounded posting lists).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


@dataclass
class TfidfConfig:
    min_df: int = 3
    max_df: int = 1500
    k: int = 50
    chunk: int = 1000


def _split(s: str) -> list[str]:
    return s.split()


def retriever_text(name_tok: pl.Series, addr_tok: pl.Series) -> list[str]:
    return [f"{a} {b}".strip() for a, b in zip(name_tok.to_list(), addr_tok.to_list())]


class TfidfIndex:
    def __init__(self, cfg: TfidfConfig | None = None):
        self.cfg = cfg or TfidfConfig()
        self.vectorizer: CountVectorizer | None = None
        self.tfidf: TfidfTransformer | None = None
        self.Xt: sparse.csr_matrix | None = None
        self.Xt_T: sparse.csr_matrix | None = None
        self.n_target: int = 0
        self.vocab: dict[str, int] = {}

    def fit(self, target_tokens: list[str]) -> dict:
        cfg = self.cfg
        cv = CountVectorizer(
            analyzer="word",
            tokenizer=_split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            min_df=cfg.min_df,
            dtype=np.float32,
        )
        Xc = cv.fit_transform(target_tokens)
        # the caller's token list is never needed again; drop it before the
        # sparse copies so peak RSS is not string-list + count matrix + slice
        del target_tokens
        raw_vocab = int(Xc.shape[1])
        df = np.asarray(Xc.sum(axis=0)).ravel()
        keep = np.flatnonzero((df >= cfg.min_df) & (df <= cfg.max_df))

        inverse = np.empty(len(cv.vocabulary_), dtype=object)
        for tok, idx in cv.vocabulary_.items():
            inverse[idx] = tok
        vocab = {tok: i for i, tok in enumerate(inverse[keep])}
        self.vocab = vocab

        X = Xc[:, keep].tocsr()
        del Xc
        tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
        self.Xt = tfidf.fit_transform(X).tocsr()
        del X
        self.Xt_T = self.Xt.T.tocsr()
        self.n_target = int(self.Xt.shape[0])
        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=_split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            vocabulary=vocab,
            dtype=np.float32,
        )
        self.tfidf = tfidf
        return {
            "raw_vocab": raw_vocab,
            "kept_vocab": int(self.Xt.shape[1]),
            "n_target": self.n_target,
            **asdict(cfg),
        }

    def transform(self, tokens: list[str]) -> sparse.csr_matrix:
        Xq = self.vectorizer.transform(tokens)
        return self.tfidf.transform(Xq).tocsr()

    def topk(self, query_tokens: list[str]) -> pl.DataFrame:
        cfg = self.cfg
        if self.Xt is None:
            raise RuntimeError("call fit() first")
        Xq = self.transform(query_tokens)
        nq = Xq.shape[0]
        sids: list[np.ndarray] = []
        tids: list[np.ndarray] = []
        ranks: list[np.ndarray] = []
        scores: list[np.ndarray] = []

        for start in range(0, nq, cfg.chunk):
            end = min(start + cfg.chunk, nq)
            S = Xq[start:end].dot(self.Xt_T).tocsr()
            for i in range(end - start):
                lo, hi = S.indptr[i], S.indptr[i + 1]
                if hi <= lo:
                    continue
                data = S.data[lo:hi]
                idx = S.indices[lo:hi]
                n = data.shape[0]
                kk = min(cfg.k, n)
                if kk < n:
                    part = np.argpartition(data, -kk)[-kk:]
                else:
                    part = np.arange(n)
                part = part[np.argsort(-data[part])]
                sids.append(np.full(kk, start + i, dtype=np.int32))
                tids.append(idx[part].astype(np.int32, copy=False))
                ranks.append(np.arange(1, kk + 1, dtype=np.int16))
                scores.append(data[part].astype(np.float32, copy=False))

        schema = {
            "qid": pl.Int32,
            "tid": pl.Int32,
            "tfidf_rank": pl.Int16,
            "tfidf_score": pl.Float32,
        }
        if not sids:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            {
                "qid": np.concatenate(sids),
                "tid": np.concatenate(tids),
                "tfidf_rank": np.concatenate(ranks),
                "tfidf_score": np.concatenate(scores),
            }
        )

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "Xt.npz", self.Xt)
        joblib.dump(
            {
                "vocab": self.vocab,
                "idf": self.tfidf.idf_,
                "cfg": asdict(self.cfg),
                "n_target": self.n_target,
            },
            d / "meta.joblib",
            compress=3,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfIndex":
        d = Path(directory)
        meta = joblib.load(d / "meta.joblib")
        obj = cls(TfidfConfig(**meta["cfg"]))
        obj.Xt = sparse.load_npz(d / "Xt.npz").tocsr()
        obj.Xt_T = obj.Xt.T.tocsr()
        obj.n_target = meta["n_target"]
        obj.vocab = dict(meta["vocab"])
        obj.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=_split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            vocabulary=meta["vocab"],
            dtype=np.float32,
        )
        tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
        tfidf.idf_ = np.asarray(meta["idf"], dtype=np.float64)
        obj.tfidf = tfidf
        return obj


class CountryRetriever:
    """One TF-IDF index per country, queried with the same country partition."""

    def __init__(self, cfg: TfidfConfig | None = None):
        self.cfg = cfg or TfidfConfig()
        self.indexes: dict[str, TfidfIndex] = {}
        self.tid_maps: dict[str, list[int]] = {}

    @staticmethod
    def _text(df: pl.DataFrame) -> list[str]:
        return retriever_text(df["name_tok"], df["addr_tok"])

    def fit(self, target: pl.DataFrame) -> dict:
        """Fit one index per country.

        Iterating over ``partition_by(as_dict=True)`` keeps every country
        partition alive for the whole loop, doubling the frame's footprint on
        top of the matrices. Filtering country-by-country keeps only one
        partition resident at a time (peaks at target + largest country).
        """
        info: dict = {}
        for key in target["country"].unique(maintain_order=True).to_list():
            part = target.filter(pl.col("country") == key)
            idx = TfidfIndex(self.cfg)
            info[key] = idx.fit(self._text(part))
            self.indexes[key] = idx
            self.tid_maps[key] = part["tid"].to_list()
            del part
        return info

    def query(self, q: pl.DataFrame) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for part in q.partition_by("country"):
            country = part["country"][0]
            idx = self.indexes.get(country)
            if idx is None:
                continue
            qsid = part["sid"].to_list()
            res = idx.topk(self._text(part))
            if res.height == 0:
                continue
            tid_map = self.tid_maps.get(country, [])
            if len(tid_map) == 0:
                continue
            res = res.with_columns(
                pl.col("qid").replace(list(range(len(qsid))), qsid).alias("sid"),
                pl.col("tid").replace(list(range(len(tid_map))), tid_map).alias("tid"),
            ).drop("qid")
            parts.append(
                res.with_columns(
                    pl.col("sid").cast(pl.Int32),
                    pl.col("tid").cast(pl.Int32),
                )
            )
        if not parts:
            return pl.DataFrame(
                schema={
                    "sid": pl.Int32,
                    "tid": pl.Int32,
                    "tfidf_rank": pl.Int16,
                    "tfidf_score": pl.Float32,
                }
            )
        return pl.concat(parts, how="vertical")

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for country, idx in self.indexes.items():
            idx.save(d / country.replace("/", "_"))
        joblib.dump(
            {"countries": list(self.indexes), "tid_maps": self.tid_maps},
            d / "countries.joblib",
            compress=3,
        )

    @classmethod
    def load(cls, directory: str | Path, cfg: TfidfConfig | None = None) -> "CountryRetriever":
        d = Path(directory)
        meta = joblib.load(d / "countries.joblib")
        obj = cls(cfg)
        for country in meta["countries"]:
            obj.indexes[country] = TfidfIndex.load(d / country.replace("/", "_"))
        obj.tid_maps = {k: list(v) for k, v in meta.get("tid_maps", {}).items()}
        if cfg is None and obj.indexes:
            obj.cfg = next(iter(obj.indexes.values())).cfg
        return obj
