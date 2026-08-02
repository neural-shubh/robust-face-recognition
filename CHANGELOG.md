# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- Corrected `.gitignore` filename (was missing the leading dot, so ignore rules were never applied by git).
- Pinned minimum dependency versions in `requirements.txt`.

## Re-ID Training Notes

- Re-ID model (custom ResNet50 backbone, trained on Market-1501) was upgraded from standard triplet sampling to
  batch-hard triplet mining (P=16 identities, K=4 images per identity per batch).
- This nearly doubled Rank-1 accuracy on the held-out evaluation set, from 39.4% to 74.4%.
- The re-ID fallback path is appearance-sensitive rather than fully identity-invariant, and is documented as such
  rather than overstated as a general disguise-invariant solution.
