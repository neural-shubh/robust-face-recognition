# Vision Transformer (ViT) Backbone Exploration

> Status: future-work / research notes. No code changes yet. Tracks issue #5.

## Motivation

The current pipeline uses CNN-based backbones (ArcFace for face embeddings, ResNet50 for re-ID). Vit (and DeiT) backbones have shown competitive or superior results on face verification and person re-ID benchmarks in recent literature, particularly under occlusion - relevant to this repo's disguise-robustness goal.

## Questions to answer before implementing

1. **Backbone choice** - ViT-B/16 vs. DeiT-S vs. a hair, vs. a hybrid CNN+Transformer (e.g. TOP-ReID-style). Pure ViTs need more data to train from scratch than we likely have - pretrained weights (ImageNet21k or face-specific) are probably required.
2. **Drop-in vs. retrain** - can we swap the ArcFace backbone alone, or does the whole embedding space need retraining against the same triplet/ArcFace loss?
3. **Compute budget** - ViTs are typically heavier at inference than MobileNetV2/ResNet50. Need a latency/accuracy tradeoff comparison, not just raw accuracy.
4. **Evaluation set** - should reuse the same benchmark setup proposed in #3 (DFW/AR Face Database) so CNN vs. ViT numbers are directly comparable.

## Proposed evaluation plan (when picked up)

- [ ] Benchmark current ArcFace/ResNet50 pipeline on the same held-out set used for #3 (baseline)
- [ ] Swap in a pretrained ViT-B/16 face embedder, evaluate zero-shot
- [ ] Fine-tune Vit on the same training set, re-evaluate
- [ ] Compare accuracy, inference latency, and model size side-by-side
- [ ] Decide go/no-go based on whether the accuracy gain justifies the compute cost

## Not in scope yet

Actual implementation and training require GPU time and the same benchmark dataset tracked in #3, so this doc is intentionally scoping-only.
