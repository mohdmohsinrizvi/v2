import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalization.text import (
    address_norm, address_numbers, leading_street_number, name_norm,
    name_numbers, name_suffix_stripped, unit_number,
)


def test_casefold_and_punct():
    assert name_norm("Orelee's  Barbershop") == "orelee s barbershop"


def test_ampersand():
    assert name_norm("Thermal & Fils SASU") == "thermal and fils sasu"


def test_accents_folded():
    assert name_norm("Café München") == "cafe munchen"


def test_suffix_expansion_and_strip():
    assert name_suffix_stripped("Golden Power Private Limited") == "golden power"
    assert name_suffix_stripped("B+ Retail Inc") == "b plus retail"


def test_unicode_devanagari_kept():
    assert "राम" in name_norm("राम मार्केटिंग")


def test_address_abbrev():
    assert address_norm("16449 Evans Rd Unit 4") == "16449 evans road unit 4"
    assert address_norm("16449 Evans Rd Apt 4") == "16449 evans road unit 4"
    assert address_norm("1795 Westchester Dr, Suite 200") == "1795 westchester drive unit 200"


def test_numbers_extracted_with_leading_zeros_normalized():
    assert address_numbers("00309 Elm St") == frozenset({309})
    assert address_numbers("006 Main") == frozenset({6})


def test_leading_street_number():
    assert leading_street_number("16449 Evans Road, Dallas") == "16449"
    assert leading_street_number("Near SBI ATM") is None


def test_unit_number():
    assert unit_number("1795 Westchester Drive, Unit 4, High Point") == "4"
    assert unit_number("1795 Westchester Drive, High Point") is None


def test_name_numbers():
    assert name_numbers("B+ Retail 24x7") == frozenset({24, 7})
