"""
Open-set rejection: calibrate similarity/distance thresholds so the
pipeline can return UNKNOWN instead of always forcing a match.

Addresses issue #4. This script does NOT hardcode a threshold — it
computes one from your own held-out known-vs-unseen identity set and
writes it to `thresholds.json`, which `open_set_gate()` below then
consumes. Run this once you have a calibration set; until then the
gate falls back to permissive defaults (effectively current behavior).

Usage:
    python threshold_calibration.py \
        --known-pairs known_pairs.csv \
        --unseen-pairs unseen_pairs.csv \
        --target-far 0.01

`known_pairs.csv` / `unseen_pairs.csv` are expected to have columns:
    img_a, img_b, stage   # stage in {"arcface", "reid"}
where known_pairs are genuine matches from your gallery and
unseen_pairs are genuinely out-of-gallery identities (per issue #4's
"held-out set of known vs. genuinely-unseen identities").

TODO(shubh): point --known-pairs / --unseen-pairs at your actual
calibration set, run this, and commit the resulting thresholds.json.
The FAR target (default 1%) is a starting point, not a fixed choice —
pick based on how costly a false accept is for your use case.
"""

import argparse
import csv
import json

import numpy as np


def load_scores(pairs_csv: str, score_fn) -> dict:
    """score_fn(img_a, img_b, stage) -> similarity/distance float.
    Left abstract here since it depends on your ArcFace / re-ID loaders
    (see extract_embedding() in the main notebook)."""
    scores = {"arcface": [], "reid": []}
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            score = score_fn(row["img_a"], row["img_b"], row["stage"])
            scores[row["stage"]].append(score)
    return scores


def calibrate_threshold(known_scores: list, unseen_scores: list, target_far: float) -> float:
    """Pick the lowest threshold such that the false-accept rate on
    unseen_scores stays at or below target_far. Assumes higher score =
    more similar (cosine similarity convention; invert for distance)."""
    unseen_scores = np.sort(np.array(unseen_scores))[::-1]
    idx = int(len(unseen_scores) * target_far)
    idx = min(max(idx, 0), len(unseen_scores) - 1)
    return float(unseen_scores[idx])


def open_set_gate(arcface_score, reid_score, thresholds: dict) -> str:
    """Returns 'MATCH' or 'UNKNOWN'. Both stages must fall below their
    calibrated threshold for a rejection, per issue #4's two-tier design."""
    arcface_reject = arcface_score < thresholds.get("arcface", -1.0)  # -1 = never reject if uncalibrated
    reid_reject = reid_score < thresholds.get("reid", -1.0)
    return "UNKNOWN" if (arcface_reject and reid_reject) else "MATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-pairs", required=True)
    parser.add_argument("--unseen-pairs", required=True)
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--out", default="thresholds.json")
    args = parser.parse_args()

    # TODO(shubh): wire this to your actual embedding extraction + cosine
    # similarity / re-ID distance functions from the main notebook.
    def score_fn(img_a, img_b, stage):
        raise NotImplementedError(
            "Plug in your ArcFace cosine-similarity / re-ID distance function here."
        )

    known = load_scores(args.known_pairs, score_fn)
    unseen = load_scores(args.unseen_pairs, score_fn)

    thresholds = {
        stage: calibrate_threshold(known[stage], unseen[stage], args.target_far)
        for stage in ("arcface", "reid")
    }

    with open(args.out, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Wrote calibrated thresholds to {args.out}: {thresholds}")
    print("Report the false-accept / false-reject rate at this threshold "
          "in the README (per issue #4), alongside the #3 benchmark numbers.")


if __name__ == "__main__":
    main()
