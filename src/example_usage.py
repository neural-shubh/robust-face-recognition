"""
Example: how to use the src/ modules end-to-end for inference
(assumes models are already downloaded — see models/README.md).

Run from the repo root: python -m src.example_usage
"""

import cv2
from functools import partial
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import tensorflow as tf

from src.models import IdentityGallery
from src.pipeline import get_face_embedding, get_reid_embedding, identify_people_with_gallery
from src.visualize import visualize_identities_named


def main():
    # --- load models ---
    person_model = YOLO("yolov8n.pt")                       # COCO-pretrained, auto-downloads
    face_model = YOLO("models/yolov8n-face.pt")              # from models/ (see download_models.py)
    face_embedder = FaceAnalysis(name="buffalo_l",
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_embedder.prepare(ctx_id=0, det_size=(640, 640))
    reid_model = tf.keras.models.load_model("models/reid_embedding_resnet50_hardmined.keras")

    # bind models into the embedding functions so IdentityGallery.enroll() has a simple signature
    face_embed_fn = partial(get_face_embedding, face_model=face_model, face_embedder=face_embedder)
    reid_embed_fn = partial(get_reid_embedding, reid_model=reid_model)

    # --- build a gallery of known people ---
    gallery = IdentityGallery(face_threshold=0.4, reid_threshold=0.5)

    known_photo = cv2.imread("path/to/known_person.jpg")
    gallery.enroll("Alice", known_photo, face_embed_fn, reid_embed_fn)

    # --- run identification on a new image ---
    identities, img = identify_people_with_gallery(
        "path/to/test_image.jpg", gallery,
        person_model, face_model, face_embedder, reid_model
    )

    # --- visualize ---
    visualize_identities_named(img, identities)


if __name__ == "__main__":
    main()
