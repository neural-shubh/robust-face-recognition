"""
The core fusion pipeline: person detection, tier-1 face identification,
tier-2 person re-ID fallback, and gallery-based name matching.

Requires these globals to be set up by the caller before use
(see notebook or example script for setup):
    - person_model   : ultralytics YOLO, COCO-pretrained (class 0 = person)
    - face_model      : ultralytics YOLO, pretrained on WIDER FACE
    - face_embedder   : insightface FaceAnalysis instance (ArcFace)
    - reid_model      : the trained TensorFlow re-ID embedding model
"""

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

FACE_CONF_THRESHOLD = 0.5


def get_reid_embedding(person_crop_bgr, reid_model):
    """Run the TF re-ID model on a person crop."""
    img = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    emb = reid_model(img, training=False).numpy()[0]
    return emb, "reid_model"


def get_face_embedding(person_crop_bgr, face_model, face_embedder,
                        conf_threshold=FACE_CONF_THRESHOLD):
    """Try YOLO face detection within a person crop; if confident, get an ArcFace embedding."""
    results = face_model.predict(person_crop_bgr, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return None, None

    best_idx = boxes.conf.argmax().item()
    conf = boxes.conf[best_idx].item()

    if conf < conf_threshold:
        return None, None

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    face_crop = person_crop_bgr[y1:y2, x1:x2]

    faces = face_embedder.get(face_crop)
    if len(faces) == 0:
        return None, None

    emb = faces[0].normed_embedding
    return emb, "face_arcface"


def identify_people(image_path, person_model, face_model, face_embedder, reid_model):
    """Basic pipeline: detect people, identify via face-first-then-re-ID, no gallery matching."""
    img = cv2.imread(image_path)
    person_results = person_model.predict(img, classes=[0], verbose=False)
    person_boxes = person_results[0].boxes

    identities = []
    for i in range(len(person_boxes)):
        x1, y1, x2, y2 = person_boxes.xyxy[i].cpu().numpy().astype(int)
        person_crop = img[y1:y2, x1:x2]

        emb, source = get_face_embedding(person_crop, face_model, face_embedder)
        if emb is None:
            emb, source = get_reid_embedding(person_crop, reid_model)

        identities.append({"bbox": (x1, y1, x2, y2), "embedding": emb, "source": source})
        print(f"Person {i}: bbox=({x1},{y1},{x2},{y2}), identified via = {source}")

    return identities, img


def identify_people_with_gallery(image_path, gallery, person_model, face_model,
                                  face_embedder, reid_model):
    """Full pipeline: detect, identify via face/re-ID, then match against a named gallery."""
    img = cv2.imread(image_path)
    person_results = person_model.predict(img, classes=[0], verbose=False)
    person_boxes = person_results[0].boxes

    identities = []
    for i in range(len(person_boxes)):
        x1, y1, x2, y2 = person_boxes.xyxy[i].cpu().numpy().astype(int)
        person_crop = img[y1:y2, x1:x2]

        emb, source = get_face_embedding(person_crop, face_model, face_embedder)
        if emb is None:
            emb, source = get_reid_embedding(person_crop, reid_model)

        name, sim = gallery.match(emb, source)
        display_name = name if name else "Unknown"

        identities.append({
            "bbox": (x1, y1, x2, y2),
            "embedding": emb,
            "source": source,
            "name": display_name,
            "similarity": sim,
        })

        print(f"Person {i}: {display_name} (sim={sim:.3f}, via={source})")

    return identities, img
