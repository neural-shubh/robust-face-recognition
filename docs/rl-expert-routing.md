# Multi-Agent Vision with RL-Based Expert Routing

> Status: exploratory research notes, tracks issue #6. No code changes yet — this is intentionally logged as a possible next architecture direction, not immediate work (per the issue itself).

## Concept recap

Instead of always running the full ensemble (face + body + motion + scene), an RL controller learns per-frame/per-track which expert(s) to consult, adapting to what's actually reliable in a given frame.

## Design sketch

### Action space (open question #1 from the issue)
Two candidate framings:
- **Discrete selection** — controller picks exactly one expert per frame. Simple, cheap, but loses the benefit of combining partial signals (e.g. face partially visible + body fully visible).
- **Soft weighting** — controller outputs a weight vector over experts, embeddings are combined via weighted fusion (similar in spirit to the existing logistic-regression meta-learner in dark-fleet-detection). More expressive, harder to train.

Leaning toward starting discrete (simpler credit assignment) and only moving to soft weighting if discrete underperforms.

### Reward signal (open question #2)
Candidates, roughly in order of implementation cost:
1. **Re-ID accuracy** — correct identity match / correct `UNKNOWN` rejection (depends on #4 landing first, since "confidently wrong" needs to be penalized more than "abstained").
2. **Confidence calibration** — reward well-calibrated confidence, not just correctness, so the controller doesn't learn to be reckless when it happens to guess right.
3. **Latency cost** — small penalty per additional expert consulted, so the controller doesn't default to always running everything.

A weighted combination of (1) and (3) is the likely starting point; (2) is a stretch goal.

### Training order (open question #3)
Two-phase, not joint:
1. Train each expert (face, body, motion, scene) independently and freeze them.
2. Train the RL controller on top of frozen experts using the reward above.

Joint training (experts + controller together) is a later idea if the two-phase approach underperforms — it's much harder to get stable.

### Proof-of-concept scope (open question #4)
Start with **2 experts only** (face + body) and a simple contextual-bandit-style controller (not full RL) before touching motion/scene experts or a real RL algorithm (PPO/DQN). This validates the routing idea cheaply before committing to the full 4-expert design.

## Dependencies on other issues
- Builds on the two-tier ArcFace → re-ID fallback already in the repo.
- Should incorporate the open-set `UNKNOWN` rejection from #4 into the reward function, so "correctly says unknown" isn't penalized the same as a wrong match.
- Benchmark numbers from #3 give a baseline (no-routing) to compare the routed system against.

## Explicitly out of scope for now
- Any actual RL training run — needs the 4 experts (or at least 2, per PoC scope) trained and frozen first.
- Choice of specific RL algorithm — deferred until the PoC validates the concept is worth pursuing at all.
