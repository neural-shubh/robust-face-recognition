"""
Threshold calibration + open-set rejection utilities (issue #4).

This module doesn't ship any calibrated numbers itself — it's the
tool to produce them. You need a labeled calibration set of:
  - known_sims:  similarity scores for genuine (same-identity) pairs,
                 e.g. re-running match() on held-out enrolled people
  - unseen_sims: similarity scores for out-of-gallery / impostor pairs,
                 e.g. match() scores for people who were never enrolled

Run this once per embedding space (face and re-ID separately, since
they live in different vector spaces per IdentityGallery's design),
then copy the resulting thresholds into src/config.py.

Example
-------
    from src.calibration import calibrate_threshold

    result = calibrate_threshold(known_sims=[...], unseen_sims=[...])
    print(result["best"])   # {"threshold": ..., "far": ..., "frr": ...}

    # Optional: plot result["sweep"] as a DET/ROC-style curve for the
    # README, per issue #3's "report FAR/FRR" ask.
"""

import numpy as np


def calibrate_threshold(known_sims, unseen_sims, num_steps=200):
    """
    Sweep candidate thresholds and pick the one minimizing FAR + FRR.

    known_sims   : similarity scores for genuine (same-identity) comparisons
    unseen_sims  : similarity scores for out-of-gallery / impostor comparisons
    num_steps    : number of threshold values to sweep between the min and
                   max observed similarity

    Returns
    -------
    dict with:
      "best"  : {"threshold": float, "far": float, "frr": float} —
                the threshold minimizing FAR + FRR
      "sweep" : list of {"threshold", "far", "frr"} across the full range,
                useful for plotting a DET/ROC-style curve
    """
    known_sims = np.asarray(known_sims, dtype=float)
    unseen_sims = np.asarray(unseen_sims, dtype=float)

    if known_sims.size == 0 or unseen_sims.size == 0:
        raise ValueError(
            "Need at least one known-identity and one unseen-identity "
            "similarity score to calibrate a threshold."
        )

    lo = float(min(known_sims.min(), unseen_sims.min()))
    hi = float(max(known_sims.max(), unseen_sims.max()))
    thresholds = np.linspace(lo, hi, num_steps)

    sweep = []
    best = None
    for t in thresholds:
        far = float((unseen_sims >= t).mean())  # unseen wrongly accepted as known
        frr = float((known_sims < t).mean())     # known wrongly rejected as unknown
        sweep.append({"threshold": float(t), "far": far, "frr": frr})
        if best is None or (far + frr) < (best["far"] + best["frr"]):
            best = {"threshold": float(t), "far": far, "frr": frr}

    return {"best": best, "sweep": sweep}


def apply_open_set_rejection(name, similarity, threshold):
    """
    Given an IdentityGallery.match() result and a calibrated threshold,
    decide whether to accept the match or return UNKNOWN.

    Kept separate from IdentityGallery.match() so the same calibrated
    threshold can be swapped in without touching pipeline.py — match()
    already returns (None, best_sim) below its *own* threshold, so this
    is mainly useful once face and re-ID stages are combined into a
    single open-set decision (per issue #4's "if both stages fall below
    their thresholds, return UNKNOWN" requirement).
    """
    if name is None or similarity < threshold:
        return None, similarity
    return name, similarity
