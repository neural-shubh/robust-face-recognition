"""
Evaluation utilities: Rank-k accuracy and mAP on the Market-1501
query/gallery protocol.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def compute_embeddings(dataset, model, batch_size=64, img_size=(256, 128)):
    paths = [s[0] for s in dataset.samples]
    person_ids = np.array([s[1] for s in dataset.samples])
    camera_ids = np.array([s[2] for s in dataset.samples])

    def load_batch(path_batch):
        imgs = []
        for p in path_batch:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = preprocess_input(img)
            imgs.append(img)
        return tf.stack(imgs)

    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = load_batch(batch_paths)
        emb = model(batch_imgs, training=False)
        embeddings.append(emb.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, person_ids, camera_ids


def evaluate_reid(query_emb, query_pids, query_cids,
                   gallery_emb, gallery_pids, gallery_cids, top_k=(1, 5, 10)):
    """
    Standard Market-1501 protocol: cosine similarity ranking, excluding
    gallery images of the same identity AND same camera as the query
    (near-duplicate crops, not genuine cross-camera re-identification).
    """
    sim_matrix = query_emb @ gallery_emb.T  # cosine sim since embeddings are L2-normalized

    num_queries = query_emb.shape[0]
    rank_hits = {k: 0 for k in top_k}
    avg_precisions = []

    for i in range(num_queries):
        sims = sim_matrix[i]
        order = np.argsort(-sims)

        q_pid, q_cid = query_pids[i], query_cids[i]

        valid_mask = ~((gallery_pids[order] == q_pid) & (gallery_cids[order] == q_cid))
        filtered_order = order[valid_mask]
        filtered_pids = gallery_pids[filtered_order]

        matches = (filtered_pids == q_pid).astype(int)
        if matches.sum() == 0:
            continue

        for k in top_k:
            if matches[:k].sum() > 0:
                rank_hits[k] += 1

        cum_matches = np.cumsum(matches)
        precision_at_i = cum_matches / (np.arange(len(matches)) + 1)
        ap = (precision_at_i * matches).sum() / matches.sum()
        avg_precisions.append(ap)

    results = {f"Rank-{k}": rank_hits[k] / num_queries for k in top_k}
    results["mAP"] = np.mean(avg_precisions)
    return results
