# Models

This directory stores the pretrained weights used by the **Disguise-Resistant Face & Person Identifier**.

The repository does **not** include model weights because GitHub has file size limits and pretrained models are distributed separately.

## Directory Structure

```text
models/
├── README.md
├── yolov8n.pt                    # YOLO Person Detector
├── yolov8n-face.pt               # YOLO Face Detector
└── reid_embedding_resnet50.keras # Trained Person Re-ID Model
```

## Download

Download the required model files from the project's Hugging Face repository and place them inside this folder.

Expected filenames:

* `yolov8n.pt`
* `yolov8n-face.pt`
* `reid_embedding_resnet50.keras`

After downloading, your directory should look like:

```text
models/
├── yolov8n.pt
├── yolov8n-face.pt
└── reid_embedding_resnet50.keras
```

## Model Description

### 1. YOLOv8 Person Detector

Detects people in an input image before person re-identification.

* Framework: Ultralytics YOLOv8
* Purpose: Person detection
* Input: RGB image
* Output: Person bounding boxes

---

### 2. YOLOv8 Face Detector

Detects visible faces inside the detected person regions.

* Framework: Ultralytics YOLOv8 Face
* Purpose: Face localization
* Input: Person crop or full image
* Output: Face bounding boxes

---

### 3. Person Re-Identification Model

Custom TensorFlow model trained on the **Market-1501** dataset.

Architecture:

* ResNet50 backbone (ImageNet pretrained)
* 256-dimensional embedding layer
* L2-normalized feature embeddings
* Batch-Hard Triplet Loss

Performance on Market-1501:

| Metric  |      Score |
| ------- | ---------: |
| Rank-1  | **74.38%** |
| Rank-5  | **88.75%** |
| Rank-10 | **92.52%** |
| mAP     | **55.32%** |

## Notes

* Keep all model filenames unchanged unless you also update the code.
* Do not commit large model files to GitHub.
* Store weights using Hugging Face Releases, GitHub Releases, or another model hosting service and download them into this directory before running the project.
