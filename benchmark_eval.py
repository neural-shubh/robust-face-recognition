"""
Quantitative benchmark evaluation on a standard disguise/occlusion dataset
(e.g. DFW - Disguised Faces in the Wild, or AR Face Database).

Addresses issue #3. This script does NOT fabricate or hardcode any accuracy
numbers - it expects a manifest of the benchmark dataset's genuine/impostor
pairs and computes TAR@FAR for the ArcFace stage alone, then again with the
re-ID fallback enabled, so the actual lift from the two-tier design (per the
project's core claim) can be reported honestly.

Usage:
    python benchmark_eval.py \
        --manifest dfw_pairs.csv \
        --far-targets 0.01,0.001

`dfw_pairs.csv` is expected to have columns:
    img_a, img_b, label, disguise_type
where label is 1 for genuine (same identity) pairs and 0 for impostor pairs,
and disguise_type is a free-text tag (e.g. "sunglasses", "cap", "beard",
"none") used to break down failure cases by disguise type per issue #3's
"which disguise types break the pipeline" ask.

TODO(shubh): point --manifest at your actual DFW / AR Face Database pairs
file (download + pair generation is not included here since it depends on
which benchmark you settle on), run this, and paste the resulting table into
the README next to the #4 threshold-calibration numbers.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def load_pairs(manifest_csv: str) -> list:
    pairs = []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            pairs.append(
                {
                    "img_a": row["img_a"],
                    "img_b": row["img_b"],
                    "label": int(row["label"]),
                    "disguise_type": row.get("disguise_type", "unknown"),
                }
            )
    return pairs


def score_pair(img_a, img_b, stage, use_reid_fallback):
    """Left abstract - depends on your ArcFace embedder / re-ID model loaders
    (see extract_embedding() in the main notebook). Should return a single
    similarity score combining ArcFace, optionally falling back to re-ID
    when use_reid_fallback=True and ArcFace confidence is low, matching the
    pipeline's actual two-tier inference logic."""
    raise NotImplementedError(
        "Plug in your ArcFace-only and ArcFace+re-ID-fallback scoring here."
    )


def tar_at_far(genuine_scores: list, impostor_scores: list, far_target: float) -> float:
    """Standard TAR@FAR: threshold set so impostor accept rate == far_target,
    then report the genuine accept rate at that threshold."""
    impostor_sorted = np.sort(np.array(impostor_scores))[::-1]
    idx = int(len(impostor_sorted) * far_target)
    idx = min(max(idx, 0), len(impostor_sorted) - 1)
    threshold = float(impostor_sorted[idx])
    tar = float(np.mean(np.array(genuine_scores) >= threshold))
    return tar


def run_condition(pairs: list, use_reid_fallback: bool, far_targets: list) -> dict:
    genuine = [p for p in pairs if p["label"] == 1]
    impostor = [p for p in pairs if p["label"] == 0]

    genuine_scores = [
        score_pair(p["img_a"], p["img_b"], "eval", use_reid_fallback) for p in genuine
    ]
    impostor_scores = [
        score_pair(p["img_a"], p["img_b"], "eval", use_reid_fallback) for p in impostor
    ]

    results = {"tar_at_far": {}, "by_disguise_type": defaultdict(list)}
    for far in far_targets:
        results["tar_at_far"][far] = tar_at_far(genuine_scores, impostor_scores, far)

    # Per-disguise-type breakdown at the first FAR target, so failure cases
    # (per issue #3) are visible rather than buried in an aggregate number.
    threshold = float(
        np.sort(np.array(impostor_scores))[::-1][
            min(max(int(len(impostor_scores) * far_targets[0]), 0), len(impostor_scores) - 1)
        ]
    )
    for p, score in zip(genuine, genuine_scores):
        results["by_disguise_type"][p["disguise_type"]].append(score >= threshold)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--far-targets", default="0.01,0.001")
    args = parser.parse_args()

    far_targets = [float(x) for x in args.far_targets.split(",")]
    pairs = load_pairs(args.manifest)

    print(f"Loaded {len(pairs)} pairs from {args.manifest}")
    print("Evaluating ArcFace-only (re-ID fallback disabled)...")
    arcface_only = run_condition(pairs, use_reid_fallback=False, far_targets=far_targets)

    print("Evaluating ArcFace + re-ID fallback (full two-tier pipeline)...")
    with_fallback = run_condition(pairs, use_reid_fallback=True, far_targets=far_targets)

    print("\n=== TAR@FAR ===")
    for far in far_targets:
        print(
            f"FAR={far}: ArcFace-only={arcface_only['tar_at_far'][far]:.4f}  "
            f"With re-ID fallback={with_fallback['tar_at_far'][far]:.4f}  "
            f"(lift={with_fallback['tar_at_far'][far] - arcface_only['tar_at_far'][far]:+.4f})"
        )

    print("\n=== Per-disguise-type accept rate (at first FAR target, with fallback) ===")
    for disguise_type, hits in sorted(with_fallback["by_disguise_type"].items()):
        print(f"{disguise_type}: {np.mean(hits):.4f} (n={len(hits)})")

    print(
        "\nCopy the TAR@FAR table and per-disguise-type breakdown above into the "
        "README's benchmark section (per issue #3) once run against a real "
        "manifest - do not hand-write numbers here."
    )


if __name__ == "__main__":
    main()
