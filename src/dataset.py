"""
Market-1501 dataset loader and triplet sampler.
"""

import os
import re
import random
from collections import defaultdict
from PIL import Image


class Market1501Dataset:
    """
    Parses Market-1501 filenames: {person_id}_{camera_seq}_{frame}_{box_idx}.jpg
    e.g. 0243_c1s1_066731_01.jpg -> person_id=243, camera=1

    Junk images (person_id == -1) are distractors/background and excluded.
    """

    def __init__(self, root_dir, split="bounding_box_train"):
        self.dir = os.path.join(root_dir, "Market-1501-v15.09.15", split)
        self.pattern = re.compile(r'(-?\d+)_c(\d)s(\d)_(\d+)_(\d+)')
        self.samples = []  # (filepath, person_id, camera_id)
        self.identity_to_indices = defaultdict(list)

        for fname in os.listdir(self.dir):
            if not fname.endswith(".jpg"):
                continue
            m = self.pattern.match(fname)
            if not m:
                continue
            person_id = int(m.group(1))
            camera_id = int(m.group(2))
            if person_id == -1:
                continue
            idx = len(self.samples)
            self.samples.append((os.path.join(self.dir, fname), person_id, camera_id))
            self.identity_to_indices[person_id].append(idx)

        self.identities = list(self.identity_to_indices.keys())
        print(f"Loaded {len(self.samples)} images, {len(self.identities)} identities from {split}")

    def __len__(self):
        return len(self.samples)

    def get_triplet(self):
        """Sample (anchor, positive, negative) for random triplet loss training."""
        anchor_id = random.choice(self.identities)
        pos_indices = self.identity_to_indices[anchor_id]
        while len(pos_indices) < 2:
            anchor_id = random.choice(self.identities)
            pos_indices = self.identity_to_indices[anchor_id]

        anchor_idx, pos_idx = random.sample(pos_indices, 2)

        neg_id = random.choice(self.identities)
        while neg_id == anchor_id:
            neg_id = random.choice(self.identities)
        neg_idx = random.choice(self.identity_to_indices[neg_id])

        return self.samples[anchor_idx], self.samples[pos_idx], self.samples[neg_idx]

    def load_image(self, path, size=(128, 64)):
        img = Image.open(path).convert("RGB")
        return img.resize(size[::-1])  # PIL uses (W, H)
