import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metric import entity_f05, macro_f05, f05


def test_perfect_match():
    assert entity_f05(["S2-1", "S3-2"], ["S2-1", "S3-2"]) == 1.0


def test_singletons_correct():
    assert entity_f05([], []) == 1.0


def test_singletons_false_merge():
    assert entity_f05(["S2-1"], []) == 0.0


def test_singletons_missed():
    assert entity_f05([], ["S2-1"]) == 0.0


def test_example_from_rules():
    # rules: pred [S2-47,S2-193,S3-812] truth [S2-47,S3-812] -> 0.714
    v = entity_f05(["S2-47", "S2-193", "S3-812"], ["S2-47", "S3-812"])
    assert abs(v - 0.714) < 0.001


def test_duplicate_pred_deduped():
    assert entity_f05(["S2-1", "S2-1"], ["S2-1"]) == 1.0


def test_macro_averages_all_entities():
    preds = {"a": ["S2-1"], "b": [], "c": ["S2-9"]}
    truths = {"a": ["S2-1"], "b": [], "c": ["S2-9"]}
    assert macro_f05(preds, truths) == 1.0


def test_macro_includes_missing_prediction_as_zero():
    preds = {"a": ["S2-1"]}
    truths = {"a": ["S2-1"], "b": ["S2-2"]}
    assert abs(macro_f05(preds, truths) - 0.5) < 1e-9


def test_f05_formula():
    assert abs(f05(2 / 3, 1.0) - 0.7142857142857143) < 1e-9
