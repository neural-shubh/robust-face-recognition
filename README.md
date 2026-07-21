# 🥷 The Disguise-Resistant Face & Person Identifier

An advanced, two-tier AI-powered computer vision pipeline designed for robust person identification and tracking under severe facial occlusions (sunglasses, caps, masks, and turned-away poses). 

---

## 🚀 Why Two Tiers?

Standard facial recognition systems instantly break down the moment a face is disguised, masked, or hidden. Rather than forcing a single model to handle impossible occlusion cases, this project splits identification into two intelligent, complementary branches:

1. **The Face Branch (Primary)**: YOLOv8n-face detects a face within a person's bounding box. If found with high confidence, an ArcFace embedding (via `insightface`) is extracted and matched against a known gallery of identities.
2. **The Person Re-ID Branch (Fallback)**: If no reliable face is visible or detectable, the system gracefully falls back to a ResNet50-based full-body appearance embedding (trained with batch-hard triplet loss on Market-1501) to match the person's posture, clothing, and build.

---

## 📐 System Pipeline Architecture

```text
Image / Live Stream → YOLOv8n (Person Detection)
                         │
                         ├── Per-Person Bounding Box Crop
                         │       │
                         │       ├── YOLOv8n-face Detection → Confidence Check
                         │       │       ├── High Confidence (≥ 0.5) → ArcFace Embedding → Face Gallery Match
                         │       │       └── Low / No Confidence ↓
                         │       └── ResNet50 Re-ID Embedding → Appearance Gallery Match
                         │
                         └── Final Annotated Output (Name, Similarity Score, Source Branch)
```

---

## 🛠️ Model Components & Stack

| Component | Model Architecture | Training Data / Source | Notes |
|---|---|---|---|
| **Person Detection** | YOLOv8n | COCO Dataset (Pretrained) | Used out-of-the-box for bounding box extraction |
| **Face Detection** | YOLOv8n-face | WIDER FACE (Pretrained) | [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) |
| **Face Embedding** | ArcFace (`buffalo_l`) | InsightFace Zoo | High-accuracy facial feature vector extraction |
| **Person Re-ID** | ResNet50 (Custom Head) | Market-1501 | Trained from scratch with batch-hard triplet loss |

---

## 📊 Re-ID Training & Optimization Results

The person re-identification branch was trained in two stages on the **Market-1501** dataset (751 identities, 12,936 training images):

| Training Stage | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---|---|---|---|
| Random Triplet Sampling | 39.4% | 63.2% | 73.5% | 22.9% |
| **+ Batch-Hard Triplet Mining (P=16, K=4)** | **74.4%** | **88.8%** | **92.5%** | **55.3%** |

> **Engineering Takeaway:** Implementing batch-hard mining (sampling the hardest positive and negative pairs within each batch) nearly doubled both Rank-1 accuracy and mean Average Precision (mAP), making the fallback branch robust enough for real-world deployment.

---

## 📦 Model Weights

Pretrained and custom-trained model weights are hosted on Hugging Face:
🔗 **[neural-shubh/disguise-resistant-face-person-id-models](https://huggingface.co/neural-shubh/disguise-resistant-face-person-id-models)**

- `reid_embedding_resnet50_hardmined.keras` — Custom TensorFlow re-ID embedding model
- `yolov8n-face.pt` — Pretrained YOLOv8n face detection weights

---

## 💻 Installation & Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/disguise-resistant-face-person-id.git
cd disguise-resistant-face-person-id
pip install -r requirements.txt
```

### 2. Run the Pipeline
Explore the complete end-to-end pipeline, model loading routines, and visualization tools inside the Jupyter notebook:
```bash
jupyter notebook face_recognition.ipynb
```

### 3. Programmatic Usage
```python
import cv2
from src.pipeline import identify_people_with_gallery
from src.models import IdentityGallery

# Initialize gallery and enroll known identities
gallery = IdentityGallery()
gallery.enroll("Alice", cv2.imread("alice_photo.jpg"))

# Run identification on a query image containing disguises/occlusions
identities, annotated_img = identify_people_with_gallery("test_image.jpg", gallery)
```

---

## ⚠️ Known Limitations & Future Work

- **Appearance vs. Identity (Re-ID):** The Re-ID branch matches clothing, posture, and build rather than invariant facial identity. Cross-session testing (changing clothes or camera angles) can degrade similarity scores. Future updates will incorporate spatiotemporal tracking to mitigate clothing dependency.
- **Hyperparameter Tuning:** Default face branch confidence thresholds (`0.5`) and gallery matching cutoffs are currently baseline estimates and require empirical calibration on validation sets.
- **Benchmark Performance:** While a 74.4% Rank-1 accuracy on Market-1501 is a strong baseline, state-of-the-art Re-ID backbones achieve higher metrics. Upgrading to stronger architectures (e.g., Vision Transformers or OSNet) is planned.

---

## 📚 Datasets
- **[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)** — Unconstrained face detection benchmark.
- **[Market-1501](https://paperswithcode.com/dataset/market-1501)** — Person re-identification dataset.

---

## 👨‍💻 Author
**Shubh Sharma**

*Tech Stack:* `ultralytics` (YOLOv8), `insightface` (ArcFace), TensorFlow / Keras, OpenCV, Google Colab (T4 GPU).
