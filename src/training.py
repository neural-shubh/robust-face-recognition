"""
Training utilities for the re-ID embedding model: losses and tf.data
pipelines for both training stages.

Stage 1 (random triplet sampling) is the baseline — simple but slow to
converge and plateaus at mediocre accuracy (Rank-1 ~39%, mAP ~23% on
Market-1501 in this project's runs).

Stage 2 (batch-hard triplet mining) fixes that by mining the hardest
positive/negative pairs within each batch instead of sampling randomly,
which nearly doubled both metrics (Rank-1 ~74%, mAP ~55%).
"""

import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (256, 128)


def load_and_preprocess(path, img_size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = preprocess_input(img)
    return img


# ---------- Stage 1: random triplet sampling ----------

def triplet_loss(margin=0.3):
    def loss_fn(anchor, positive, negative):
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss_fn


def triplet_generator(market_dataset, num_triplets=5000):
    def gen():
        for _ in range(num_triplets):
            a, p, n = market_dataset.get_triplet()
            yield a[0], p[0], n[0]
    return gen


def make_triplet_dataset(market_dataset, num_triplets=5000, batch_size=32, img_size=IMG_SIZE):
    ds = tf.data.Dataset.from_generator(
        triplet_generator(market_dataset, num_triplets),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )
    ds = ds.map(
        lambda a, p, n: (
            load_and_preprocess(a, img_size),
            load_and_preprocess(p, img_size),
            load_and_preprocess(n, img_size),
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_train_step(model, loss_fn, optimizer):
    @tf.function
    def train_step(a_batch, p_batch, n_batch):
        with tf.GradientTape() as tape:
            emb_a = model(a_batch, training=True)
            emb_p = model(p_batch, training=True)
            emb_n = model(n_batch, training=True)
            loss = loss_fn(emb_a, emb_p, emb_n)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return train_step


# ---------- Stage 2: batch-hard triplet mining ----------

def pk_batch_generator(market_dataset, P=16, K=4, num_batches=300):
    """P identities per batch, K images per identity -> batch size P*K."""
    eligible_ids = [pid for pid, idxs in market_dataset.identity_to_indices.items() if len(idxs) >= K]

    def gen():
        for _ in range(num_batches):
            batch_ids = random.sample(eligible_ids, P)
            batch_paths, batch_labels = [], []
            for pid in batch_ids:
                idxs = random.sample(market_dataset.identity_to_indices[pid], K)
                for idx in idxs:
                    path, person_id, cam_id = market_dataset.samples[idx]
                    batch_paths.append(path)
                    batch_labels.append(person_id)
            yield batch_paths, batch_labels
    return gen


def make_pk_dataset(market_dataset, P=16, K=4, num_batches=300, img_size=IMG_SIZE):
    gen = pk_batch_generator(market_dataset, P, K, num_batches)

    def wrapped_gen():
        for paths, labels in gen():
            imgs = tf.stack([load_and_preprocess(p, img_size) for p in paths])
            yield imgs, tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_generator(
        wrapped_gen,
        output_signature=(
            tf.TensorSpec(shape=(P * K, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(P * K,), dtype=tf.int32),
        )
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
    square_norm = tf.linalg.diag_part(dot_product)
    dist_matrix = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    dist_matrix = tf.maximum(dist_matrix, 0.0)

    labels = tf.reshape(labels, [-1, 1])
    same_identity_mask = tf.equal(labels, tf.transpose(labels))
    diff_identity_mask = tf.logical_not(same_identity_mask)

    identity_mask_no_self = tf.logical_and(
        same_identity_mask, tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    )
    pos_dist = tf.where(identity_mask_no_self, dist_matrix, -1.0)
    hardest_positive = tf.reduce_max(pos_dist, axis=1)

    max_dist = tf.reduce_max(dist_matrix)
    neg_dist = tf.where(diff_identity_mask, dist_matrix, max_dist + 1.0)
    hardest_negative = tf.reduce_min(neg_dist, axis=1)

    loss = tf.maximum(hardest_positive - hardest_negative + margin, 0.0)
    return tf.reduce_mean(loss)


def make_train_step_hard(model, optimizer, margin=0.3):
    @tf.function
    def train_step_hard(imgs, labels):
        with tf.GradientTape() as tape:
            embeddings = model(imgs, training=True)
            loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return train_step_hard


def train(model, dataset, train_step_fn, epochs=15, hard_mining=False):
    """Generic training loop for either stage — pass the matching train_step function."""
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for batch in dataset:
            if hard_mining:
                imgs, labels = batch
                loss = train_step_fn(imgs, labels)
            else:
                a_batch, p_batch, n_batch = batch
                loss = train_step_fn(a_batch, p_batch, n_batch)
            epoch_loss.update_state(loss)
        loss_name = "batch-hard triplet loss" if hard_mining else "triplet loss"
        print(f"Epoch {epoch+1}/{epochs} - {loss_name}: {epoch_loss.result():.4f}")
