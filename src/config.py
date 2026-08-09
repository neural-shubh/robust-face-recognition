"""
Central config for gallery-matching thresholds (issue #4).

These were previously hardcoded as constructor defaults inside
IdentityGallery (see src/models.py). Pulling them out here means they
can be re-tuned per deployment/dataset without touching pipeline code,
and gives an obvious place to plug in the output of
src/calibration.calibrate_threshold() once you have a labeled
calibration set.

Current values are still the original untuned defaults called out in
the README's "Known limitations" section — not yet calibrated against
held-out known vs. genuinely-unseen identities.
"""

FACE_MATCH_THRESHOLD = 0.4
REID_MATCH_THRESHOLD = 0.5
