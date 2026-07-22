"""
Model definitions: the re-ID embedding network and the identity gallery
used for matching detected people against known identities.
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


def build_reid_embedding_model(embedding_dim=256, input_shape=(256, 128, 3)):
    """
    ResNet50 backbone (ImageNet-pretrained) with a custom embedding head,
    L2-normalized output for cosine-similarity-based matching.
    """
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape)
    x = base.output
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inputs=base.input, outputs=x, name="reid_embedding_net")


class IdentityGallery:
    """
    Stores known identities with both face and re-ID embeddings.

    Face embeddings (ArcFace, 512-dim) and re-ID embeddings (this project's
    ResNet50 model, 256-dim) live in different vector spaces and are kept
    in separate galleries — matching only ever compares like with like.
    """

    def __init__(self, face_threshold=0.4, reid_threshold=0.5):
        self.face_gallery = {}   # name -> list of face embeddings
        self.reid_gallery = {}   # name -> list of re-id embeddings
        self.face_threshold = face_threshold
        self.reid_threshold = reid_threshold

    def enroll(self, name, person_crop_bgr, get_face_embedding_fn, get_reid_embedding_fn):
        """
        Add a person to the gallery. Tries to enroll both embedding types
        from the same crop when possible, so this person can be matched
        later whether their face is visible or not.
        """
        face_emb, _ = get_face_embedding_fn(person_crop_bgr)
        if face_emb is not None:
            self.face_gallery.setdefault(name, []).append(face_emb)
            print(f"Enrolled '{name}' — face embedding added")

        reid_emb, _ = get_reid_embedding_fn(person_crop_bgr)
        self.reid_gallery.setdefault(name, []).append(reid_emb)
        print(f"Enrolled '{name}' — re-ID embedding added")

    def _cosine_sim(self, emb, gallery_embs):
        gallery_matrix = np.stack(gallery_embs)
        sims = gallery_matrix @ emb / (
            np.linalg.norm(gallery_matrix, axis=1) * np.linalg.norm(emb) + 1e-8
        )
        return sims.max()

    def match(self, emb, source):
        """
        Match an embedding against the correct gallery based on source
        ('face_arcface' or 'reid_model'). Returns (name, similarity) —
        name is None if no match clears the threshold.
        """
        if source == "face_arcface":
            gallery, threshold = self.face_gallery, self.face_threshold
        else:
            gallery, threshold = self.reid_gallery, self.reid_threshold

        if not gallery:
            return None, 0.0

        best_name, best_sim = None, -1.0
        for name, embs in gallery.items():
            sim = self._cosine_sim(emb, embs)
            if sim > best_sim:
                best_name, best_sim = name, sim

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def save(self, path="identity_gallery.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"face": self.face_gallery, "reid": self.reid_gallery}, f)
        print(f"Gallery saved to {path}")

    def load(self, path="identity_gallery.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.face_gallery = data["face"]
        self.reid_gallery = data["reid"]
        print(f"Gallery loaded: {len(self.face_gallery)} face identities, "
              f"{len(self.reid_gallery)} re-id identities")
