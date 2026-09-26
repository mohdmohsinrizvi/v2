"""Row-level normalization for business names and addresses.

Produces the canonical normalized schema used by every downstream stage.
Everything here is pure-Python-safe and Unicode/Devanagari aware (see
``src/normalization/text.py`` for the character-level rules).
"""
from __future__ import annotations

import re

import polars as pl

from src.normalization.text import _NUM_RE, address_parts, name_parts

FIELDS = [
    ("name_norm", pl.Utf8),
    ("name_ss", pl.Utf8),
    ("name_tok", pl.Utf8),
    ("name_num", pl.Utf8),
    ("name_nchar", pl.Int32),
    ("addr_norm", pl.Utf8),
    ("addr_tok", pl.Utf8),
    ("addr_num", pl.Utf8),
    ("addr_nchar", pl.Int32),
    ("street_num", pl.Utf8),
    ("unit_num", pl.Utf8),
    ("bldg_num", pl.Utf8),
    ("postal", pl.Utf8),
    ("has_addr", pl.Boolean),
]

STRUCT_DTYPE = pl.Struct(dict(FIELDS))

_NUMS_JOIN = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")

# Post-normalization word sets (pre-expansion spellings like "apt"/"rd" are
# already expanded by address_parts, so both forms are listed for safety).
_UNIT_WORDS = frozenset({"unit", "suite", "ste", "apt", "flat", "no",
                         "number", "floor", "bldg", "building"})
_BLDG_WORDS = frozenset({"bldg", "building", "tower", "block"})


def _addr_meta(tokens: list[str]) -> tuple[str, str, str]:
    """Pull (unit, building, postal) out of an address token list in one pass.

    Equivalent to running the three original regexes independently over the
    normalized address, but ~6x faster.
    """
    unit = bldg = postal = ""
    n = len(tokens)
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < n else ""
        if not unit and tok in _UNIT_WORDS and nxt.isdecimal():
            unit = nxt
        if not bldg and tok in _BLDG_WORDS and nxt.isascii() and nxt.isalnum():
            bldg = nxt
        if not postal and tok.isdecimal() and 4 <= len(tok) <= 6:
            postal = tok
    return unit, bldg, postal


def _nums(values) -> str:
    return ",".join(str(v) for v in sorted(set(values)))


def normalize_row(name: str | None, addr: str | None) -> dict:
    nn, ss, toks = name_parts(name or "")
    abase, an, atoks = address_parts(addr or "")
    anums = [int(x) for x in _NUM_RE.findall(an)]

    street = ""
    for t in abase.split()[:3]:
        if t.isdigit():
            street = str(int(t))
            break

    m_unit, m_bldg, postal = _addr_meta(atoks)

    return {
        "name_norm": nn,
        "name_ss": ss,
        "name_tok": " ".join(toks),
        "name_num": _nums(int(x) for x in _NUM_RE.findall(nn)),
        "name_nchar": len(nn),
        "addr_norm": an,
        "addr_tok": " ".join(atoks),
        "addr_num": _nums(anums),
        "addr_nchar": len(an),
        "street_num": street,
        "unit_num": m_unit,
        "bldg_num": m_bldg,
        "postal": postal,
        "has_addr": bool(an),
    }


def normalize_df(df: pl.DataFrame, batched: bool = True) -> pl.DataFrame:
    """Add all normalized columns to a raw source frame."""
    expr = pl.struct(["business_name", "business_address"]).map_batches(
        _normalize_batch, return_dtype=STRUCT_DTYPE
    ).alias("__norm")
    out = df.with_columns(expr)
    names = [f for f, _ in FIELDS]
    return out.unnest("__norm").select(
        [c for c in out.columns if c != "__norm"] + names
    )


def _normalize_batch(s: pl.Series) -> pl.Series:
    """Vectorized-ish batch normalizer (one Python call per batch)."""
    struct = s.struct.unnest()
    names = struct["business_name"].to_list()
    addrs = struct["business_address"].to_list()
    rows = [normalize_row(n, a) for n, a in zip(names, addrs)]
    frame = pl.DataFrame(rows, schema=FIELDS) if rows else pl.DataFrame(schema=FIELDS)
    return frame.select(pl.struct([f for f, _ in FIELDS])).to_series()
