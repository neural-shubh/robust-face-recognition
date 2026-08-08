# Multi-Agent Vision with RL-Based Expert Routing — Design Notes

Tracking issue: #6. This is exploratory architecture planning, not
implementation — logged per the issue's own framing ("track as a possible
next architecture direction rather than an immediate implementation task").

## Problem restated
Right now the pipeline always runs the full ArcFace → re-ID fallback chain.
A learned controller could instead decide, per-track, which expert(s) are
worth consulting given what's actually visible.

## Proposed phased approach (per issue checklist)

### Phase 0 — Proof of concept (2 experts, bandit, not full RL)
- Experts: **face** (existing ArcFace embedder) and **body** (existing re-ID ResNet50)
- Controller: contextual multi-armed bandit, not a full RL policy — much
  simpler to train and debug, and the issue itself suggests starting here
- Context features for the bandit: face-detector confidence, face bounding-box
  size/visibility ratio, whether YOLOv8n-face returned a detection at all
- Action space: {face-only, body-only, both (current default)}
- Reward: correct top-1 match on a held-out labeled set, minus a small
  penalty for running both experts (proxy for the latency cost)

### Phase 1 — Add motion/scene experts, evaluate whether bandit still suffices
Only move to a full RL formulation (state = rolling track history, not just
current frame) if the bandit's context space stops being expressive enough
— e.g. if the best expert choice depends on *sequences* of frames rather
than a single frame's visibility cues.

## Open questions carried over from the issue (unresolved, need a decision before Phase 0 starts)
- [ ] Discrete expert selection vs. soft weighting of expert outputs
- [ ] Reward signal: pure accuracy vs. accuracy + calibration vs. accuracy + latency cost
- [ ] Whether experts are frozen (already trained) during controller training, or fine-tuned jointly — freezing is simpler and recommended for Phase 0

## Relationship to other open issues
- Depends conceptually on #4 (open-set rejection) — a router that can pick
  "consult no expert, return UNKNOWN" is a natural extension once thresholded
  rejection exists for each individual expert
- Independent of #5 (ViT backbone) — backbone choice for each expert is
  orthogonal to how the controller routes between experts

## Not in scope for Phase 0
- Scene expert (contextual/environmental cues) — no existing model in this
  repo produces that signal yet, would need to be built from scratch first
- Full RL controller — see Phase 1 gate above
