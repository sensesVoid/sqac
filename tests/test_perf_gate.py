"""Performance gate tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments.perf_gate import check, check_median


def test_check_passes_within_threshold():
    baseline = {"size": 5000, "build_s": 30.0, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    current = {"size": 5000, "build_s": 32.0, "exact_ms": 5.5, "fuzzy_hit_ms": 10.5, "fuzzy_miss_ms": 7.5}
    ok, report = check(baseline, current, threshold=0.20)
    assert ok is True
    assert "PASSED" in report


def test_check_fails_on_regression():
    baseline = {"size": 5000, "build_s": 30.0, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    current = {"size": 5000, "build_s": 40.0, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    ok, report = check(baseline, current, threshold=0.20)
    assert ok is False
    assert "build_s" in report
    assert "FAILED" in report


def test_check_median_of_three():
    baseline = {"size": 5000, "build_s": 30.0, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    currents = [
        {"size": 5000, "build_s": 28.0, "exact_ms": 4.8, "fuzzy_hit_ms": 9.5, "fuzzy_miss_ms": 6.8},
        {"size": 5000, "build_s": 40.0, "exact_ms": 5.2, "fuzzy_hit_ms": 10.2, "fuzzy_miss_ms": 7.1},
        {"size": 5000, "build_s": 31.0, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0},
    ]
    ok, report = check_median(baseline, currents, threshold=0.20)
    assert ok is True


def test_check_load_s_regression():
    baseline = {"size": 5000, "build_s": 30.0, "load_s": 0.1, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    current = {"size": 5000, "build_s": 30.0, "load_s": 0.5, "exact_ms": 5.0, "fuzzy_hit_ms": 10.0, "fuzzy_miss_ms": 7.0}
    ok, report = check(baseline, current, threshold=0.20)
    assert ok is False
    assert "load_s" in report