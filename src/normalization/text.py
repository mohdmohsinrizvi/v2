from __future__ import annotations

import re
import unicodedata
from typing import Iterable

LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "sarl", "sas", "sasu", "eurl",
    "sa", "sci", "srl", "gmbh", "bv", "nv", "plc", "pty", "traders", "trading",
    "enterprises", "enterprise", "group", "services", "service", "solutions",
    "and", "associates", "partners", "holdings", "industries", "consultants",
    "consulting", "agency", "stores", "store", "shop", "centre", "center",
}

SUFFIX_EXPAND = {
    "pvt": "private", "ltd": "limited", "inc": "incorporated",
    "corp": "corporation", "co": "company", "llc": "llc", "llp": "llp",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "sci": "sci", "pty": "pty", "plc": "plc", "srl": "srl",
}

ADDRESS_EXPAND = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "apartment": "unit", "apt": "unit", "ste": "unit", "suite": "unit",
    "flat": "unit", "number": "unit", "no": "unit",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "fl": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
}

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")
_UNIT_RE = re.compile(r"\b(unit|suite|ste|apt|flat|no|number|floor|bldg|building)\b\.?\s*#?\s*(\d+)\b", re.I)
_BUILDING_RE = re.compile(r"\b(bldg|building|tower|block)\b\.?\s*#?\s*([A-Za-z0-9]+)\b", re.I)

# Combining marks that belong to Latin/Greek/Cyrillic/etc. diacritical blocks.
# Devanagari (and other Indic) matras live in their own script blocks, so they are
# deliberately NOT matched here and survive normalization intact.
_COMBINING_RE = re.compile(
    r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20f0\ufe20-\ufe2f]+"
)

# Single-pass "collapse to single space" class: whitespace + P/S/C characters.
# Covers all ASCII punctuation, Latin-1 punctuation/symbols, general punctuation,
# currency, arrows/symbols, box drawing, CJK punctuation, fullwidth forms, emoji,
# and the Devanagari danda. Characters outside these ranges are letters/digits in
# scripts we must preserve (Devanagari, CJK, accents), so they are kept.
_CLEAN_RE = re.compile(
    r"[\s\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e"
    r"\u00a1-\u00bf\u037e\u0387"
    r"\u055a-\u055f\u0589-\u058a"
    r"\u0609-\u060d\u061b\u061e\u061f\u066a-\u066d\u06d4"
    r"\u0964\u0965"
    r"\u2010-\u2027\u2030-\u205e\u20a0-\u20bf\u2100-\u214f"
    r"\u2190-\u24ff\u2500-\u27bf\u2983-\u2998"
    r"\u2e00-\u2e7f\u3000-\u303f"
    r"\ufe10-\ufe6b\uff01-\uff65\U0001f300-\U0001faff]+"
)


def _fold(s: str) -> str:
    """NFKD + drop Latin/Greek/Cyrillic diacritics + casefold.

    Combining marks are only removed for the diacritical blocks listed in
    ``_COMBINING_RE``; Indic matras (e.g. Devanagari) live in their own script
    blocks and are preserved, which the unit tests enforce.
    """
    s = unicodedata.normalize("NFKD", s)
    s = _COMBINING_RE.sub("", s)
    return s.casefold()


def _strip_punct(s: str) -> str:
    r"""Collapse whitespace and punctuation/symbol/control runs to one space.

    Deliberately NOT ``[^\w\s]``: Python's ``\\w`` misses Unicode spacing marks
    (e.g. Devanagari matras, category Mc), which would silently destroy non-Latin
    names. The class below whitelists the P/S/C ranges instead of blacklisting
    everything that is not a word character.
    """
    return _CLEAN_RE.sub(" ", s)


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = _fold(s)
    s = s.replace("&", " and ")
    s = s.replace("+", " plus ")
    s = _strip_punct(s)
    return s.strip()



def expand_suffixes(tokens: Iterable[str]) -> list[str]:
    return [SUFFIX_EXPAND.get(t, t) for t in tokens]


def name_parts(raw: str) -> tuple[str, str, list[str]]:
    """Single-pass ``(name_norm, name_suffix_stripped, tokens)``.

    Equivalent to calling the three individual helpers, but normalizes the raw
    string only once (this sits on the hot path for ~20M rows).
    """
    base = normalize_text(raw)
    toks = expand_suffixes(base.split())
    hi = len(toks)
    lo = 0
    while hi > lo and toks[hi - 1] in LEGAL_SUFFIXES:
        hi -= 1
    while lo < hi and toks[lo] in LEGAL_SUFFIXES:
        lo += 1
    return base, " ".join(toks[lo:hi]), toks


def address_parts(raw: str | None) -> tuple[str, str, list[str]]:
    """Single-pass ``(base_normalized, address_norm, tokens)``."""
    base = normalize_text(raw)
    toks = [ADDRESS_EXPAND.get(t, t) for t in base.split()]
    return base, " ".join(toks), toks


def name_norm(raw: str) -> str:
    return normalize_text(raw)


def name_suffix_stripped(raw: str) -> str:
    return name_parts(raw)[1]


def name_tokens(raw: str) -> list[str]:
    return name_parts(raw)[2]


def name_numbers(raw: str) -> frozenset[int]:
    return frozenset(int(x) for x in _NUM_RE.findall(normalize_text(raw)))


def address_norm(raw: str | None) -> str:
    return address_parts(raw)[1]


def address_tokens(raw: str | None) -> list[str]:
    return address_parts(raw)[2]



def address_numbers(raw: str | None) -> frozenset[int]:
    return frozenset(int(x) for x in _NUM_RE.findall(address_norm(raw)))


def strip_leading_zeros(nums: Iterable[int]) -> frozenset[int]:
    return frozenset(int(str(abs(int(n)))) for n in nums)


def unit_number(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _UNIT_RE.search(address_norm(raw))
    return m.group(2) if m else None


def building_number(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _BUILDING_RE.search(address_norm(raw))
    return m.group(2) if m else None


def leading_street_number(raw: str | None) -> str | None:
    if not raw:
        return None
    for t in normalize_text(raw).split()[:3]:
        if t.isdigit():
            return str(int(t))
    return None
