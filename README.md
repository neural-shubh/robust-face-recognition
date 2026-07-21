# Disguise-Resistant Face & Person Identifier

A two-tier identification pipeline that stays robust when faces are partially or fully occluded (sunglasses, masks, caps, turned-away poses). When a face is visible and detectable, it identifies people using face recognition. When it isn't, it falls back to person re-identification based on appearance (clothing, build, posture).

## Why two tiers?

Face recognition alone breaks down the moment a face is disguised or hidden. Rather than trying to force a single model to be robust to every occlusion case, this project splits identification into two independent branches and picks whichever one has usable signal:

1. **Face branch** — YOLOv8n-face detects a face within a person's bounding box; if found with sufficient confidence, an ArcFace (via `insightface`) embedding is extracted and matched against a gallery of known identities.
2. **Person re-ID branch** — if no reliable face is found, a ResNet50-based embedding (trained with batch-hard triplet loss on Market-1501) matches the person's full-body appearance against the gallery instead.

## Pipeline

```
Image → YOLOv8n (person detection)
           │
           ├── per person crop
           │       │
           │       ├── YOLOv8n-face detection → confidence check
           │       │       ├── high confidence → ArcFace embedding → face gallery match
           │       │       └── low/no confidence ↓
           │       └── ResNet50 re-ID embedding → re-ID gallery match
           │
           └── annotated output (name, similarity score, source branch)
```

## Models

| Component | Model | Training data | Notes |
|---|---|---|---|
| Person detection | YOLOv8n | COCO (pretrained) | Used as-is, no fine-tuning |
| Face detection | YOLOv8n-face | WIDER FACE (pretrained) | [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) |
| Face embedding | ArcFace (buffalo_l) | — | Via `insightface`, used as-is |
| Person re-ID embedding | ResNet50 (custom head) | Market-1501 | Trained from scratch here with batch-hard triplet loss |

Pretrained weights for person/face detection and face embedding are used directly — the effort here went into the re-ID branch (trained specifically for this project) and the fusion/gallery-matching logic that ties both branches together.

## Re-ID training results

Trained in two stages on Market-1501 (751 identities, 12,936 training images):

| Stage | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---|---|---|---|
| Random triplet sampling | 39.4% | 63.2% | 73.5% | 22.9% |
| + Batch-hard triplet mining (P=16, K=4) | **74.4%** | **88.8%** | **92.5%** | **55.3%** |

Batch-hard mining — sampling the hardest positive/negative pairs within each batch rather than random triplets — nearly doubled both Rank-1 accuracy and mAP. This is the single biggest lever in re-ID training and is the reason the fallback branch is usable at all.

## Weights

Pretrained/trained model weights are hosted on Hugging Face (kept out of this repo due to size):
**[neural-shubh/disguise-resistant-face-person-id-models](https://huggingface.co/neural-shubh/disguise-resistant-face-person-id-models)**

- `reid_embedding_resnet50_hardmined.keras` — TensorFlow re-ID embedding model
- `yolov8n-face.pt` — pretrained YOLOv8n face detector

## Usage

```python
from src.pipeline import identify_people_with_gallery
from src.models import IdentityGallery

gallery = IdentityGallery()
gallery.enroll("Alice", cv2.imread("alice_photo.jpg"))

identities, img = identify_people_with_gallery("test_image.jpg", gallery)
```

See `face_recognition.ipynb` for the full end-to-end pipeline, including model loading, training code, and visualization.

## Known limitations

- **Re-ID is appearance-sensitive, not identity-invariant.** It matches clothing, build, and posture — not a person's underlying identity the way a human would. Testing showed that comparing enrollment and query photos taken in different sessions (different clothes, framing, or background) drops similarity scores well below the match threshold, even for the same person. A controlled same-session test (same clothes, only the face occlusion changing) is needed to isolate disguise-robustness specifically from appearance drift.
- **Face branch confidence threshold (0.5) and gallery match thresholds are untuned defaults**, not calibrated against a labeled validation set — expect to adjust for your own use case.
- **Re-ID Rank-1 (74%) and mAP (55%) are a solid baseline but below published SOTA** (~90%+ Rank-1 on Market-1501 with stronger backbones and longer training).

## Datasets

- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) — face detection
- [Market-1501](https://paperswithcode.com/dataset/market-1501) — person re-identification

## Tech stack

`ultralytics` (YOLOv8), `insightface` (ArcFace), TensorFlow/Keras (re-ID model), OpenCV, Google Colab (T4 GPU)
