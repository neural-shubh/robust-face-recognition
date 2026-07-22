# Data

This project uses two public datasets. Neither is included in this repo (too large for git) — download them with the snippets below, into this `data/` directory.

## Directory structure (after download)

```
data/
├── wider_face/
│   └── WIDER Face Dataset For YOLOv12/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           └── images/
├── market1501/
│   └── Market-1501-v15.09.15/
│       ├── bounding_box_train/
│       ├── bounding_box_test/
│       ├── query/
│       ├── gt_bbox/
│       └── gt_query/
└── README.md   (this file)
```

## 1. WIDER FACE (face detection)

Used to evaluate/fine-tune the YOLOv8 face detector. Pre-converted to YOLO label format (bounding boxes in `.txt` files matching each image).

- **Source:** [canomercik/wider-face-dataset-for-yolov12-format](https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format) (Kaggle)
- **Original dataset:** [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
- **Size:** ~3.4 GB, 32,203 images, 393,703 labeled faces
- **Split:** train (12,880 images) / val (3,226 images) / test (16,097 images, unlabeled)

```python
import kagglehub
path = kagglehub.dataset_download("canomercik/wider-face-dataset-for-yolov12-format")
```

**Note:** this project uses a pretrained YOLOv8n-face detector directly (see `models/README.md`), so re-downloading/re-training on WIDER FACE is optional — only needed if you want to fine-tune the face detector yourself.

## 2. Market-1501 (person re-identification)

Used to train and evaluate the person re-ID embedding model (the fallback branch for occluded/disguised faces).

- **Source:** [pengcw1/market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501) (Kaggle)
- **Original dataset:** [Market-1501](https://paperswithcode.com/dataset/market-1501)
- **Size:** ~146 MB, 1,501 identities across 6 camera views
- **Splits:**
  - `bounding_box_train/` — 12,936 images, 751 identities (training)
  - `bounding_box_test/` — 15,913 images, 751 identities (gallery, for evaluation)
  - `query/` — 3,368 images, 750 identities (queries, for evaluation)
  - `gt_bbox/`, `gt_query/` — ground-truth annotations for the official evaluation protocol

```python
import kagglehub
path = kagglehub.dataset_download("pengcw1/market-1501")
```

**Filename convention:** `{person_id}_{camera}s{sequence}_{frame}_{box_idx}.jpg`
e.g. `0243_c1s1_066731_01.jpg` → person ID 243, camera 1. `person_id == -1` marks junk/distractor images and is excluded during loading (handled automatically by `Market1501Dataset` in the notebook).

## Notes

- Both datasets download via `kagglehub`, which requires a Kaggle account and API token (`~/.kaggle/kaggle.json`). See [Kaggle's API docs](https://www.kaggle.com/docs/api) if you don't have one set up.
- Paths above assume Google Colab's default `kagglehub` cache location (`/root/.cache/kagglehub/datasets/...`). Adjust paths in the notebook if running elsewhere (e.g. locally, or Kaggle Notebooks' `/kaggle/input/` convention).
- No personal or identity-enrollment data (e.g. your own photos used with `IdentityGallery.enroll()`) is included here — that's generated locally per user and intentionally excluded via `.gitignore`.
