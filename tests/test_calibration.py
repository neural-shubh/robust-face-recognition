"""
Unit tests for src/calibration.py (issue #7).

Pure numpy logic, synthetic data only -- no model weights, no GPU,
no network access required.
"""

import numpy as np
import pytest

from src.calibration import apply_open_set_rejection, calibrate_threshold


class TestCalibrateThreshold:
    def test_picks_sane_threshold_on_separated_distributions(self):
        # Known (genuine) pairs cluster high, unseen (impostor) pairs cluster low.
        rng = np.random.default_rng(0)
        known_sims = rng.normal(loc=0.85, scale=0.02, size=200)
        unseen_sims = rng.normal(loc=0.30, scale=0.02, size=200)

        result = calibrate_threshold(known_sims, unseen_sims)

        assert "best" in result and "sweep" in result
        best = result["best"]
        # The chosen threshold should sit clearly between the two clusters.
        assert 0.30 < best["threshold"] < 0.85
        # With well-separated distributions, FAR and FRR at the best point
        # should both be near zero.
        assert best["far"] < 0.05
        assert best["frr"] < 0.05

    def test_sweep_length_matches_num_steps(self):
        known_sims = [0.9, 0.8, 0.85]
        unseen_sims = [0.1, 0.2, 0.15]
        result = calibrate_threshold(known_sims, unseen_sims, num_steps=50)
        assert len(result["sweep"]) == 50

    def test_raises_on_empty_known_sims(self):
        with pytest.raises(ValueError):
            calibrate_threshold(known_sims=[], unseen_sims=[0.1, 0.2])

    def test_raises_on_empty_unseen_sims(self):
        with pytest.raises(ValueError):
            calibrate_threshold(known_sims=[0.9, 0.8], unseen_sims=[])

    def test_raises_on_both_empty(self):
        with pytest.raises(ValueError):
            calibrate_threshold(known_sims=[], unseen_sims=[])


class TestApplyOpenSetRejection:
    def test_accepts_match_at_or_above_threshold(self):
        name, sim = apply_open_set_rejection("alice", similarity=0.7, threshold=0.5)
        assert name == "alice"
        assert sim == 0.7

    def test_accepts_match_exactly_at_threshold(self):
        name, sim = apply_open_set_rejection("bob", similarity=0.5, threshold=0.5)
        assert name == "bob"

    def test_rejects_match_below_threshold(self):
        name, sim = apply_open_set_rejection("carol", similarity=0.4, threshold=0.5)
        assert name is None
        assert sim == 0.4

    def test_rejects_when_name_is_none_regardless_of_similarity(self):
        name, sim = apply_open_set_rejection(None, similarity=0.99, threshold=0.5)
        assert name is None
        assert sim == 0.99
