# Vision Transformer (ViT) Backbone — Exploration Notes

Tracking issue: #5. This is a research-planning doc, not an implementation —
no code changes to the pipeline are proposed here yet.

## Why consider it
The current pipeline uses YOLOv8n (detection) + ArcFace/ResNet50 (embeddings),
all CNN-based. ViTs have shown competitive or better accuracy on face
recognition benchmarks (e.g. ViT-Face, EVA-Face lineage) and, more relevant
to this repo's disguise-resistance goal, self-attention can in principle
weight visible (undisguised) facial regions more heavily than occluded ones
— a CNN's local receptive fields don't get that for free.

## Candidate backbones to benchmark
| Backbone | Params | Pretrained on | Notes |
|---|---|---|---|
| ViT-B/16 | 86M | ImageNet-21k | baseline, well-supported in `timm` |
| DeiT-S | 22M | ImageNet-1k (distilled) | much cheaper, good if compute-limited |
| Swin-T | 28M | ImageNet-1k | hierarchical, may suit re-ID better than plain ViT |

## What a fair comparison needs
- [ ] Same training data / augmentation as the current ArcFace + ResNet50 re-ID setup (Market-1501 for re-ID, existing face dataset for ArcFace) — swapping backbone AND dataset at once makes results uninterpretable
- [ ] Same loss (triplet / ArcFace-margin) so the comparison isolates backbone choice
- [ ] Report accuracy AND inference latency/memory — issue is explicit that compute cost is a real tradeoff, not just accuracy
- [ ] Evaluate specifically on occluded/disguised subsets (not just clean-face accuracy), since that's this repo's actual value proposition

## Honest expectation setting
ViTs typically need more data or stronger augmentation/distillation to match CNNs at this model scale — DeiT's whole premise is data-efficient distillation to get around that. Given this repo trains on Market-1501 (not huge), a naive ViT swap may *underperform* the current ResNet50 without careful regularization. That's a real risk to flag before sinking time in, not a reason to skip the experiment.

## Suggested first step
Small proof-of-concept: swap only the re-ID branch to DeiT-S (cheapest option), keep ArcFace stage untouched, and compare Rank-1/mAP against the current 74.4%/55.3% baseline on the same Market-1501 eval split. If that doesn't show a lift, larger ViT-B is unlikely to be worth the added cost for this dataset size.
