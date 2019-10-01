import random
import numpy as np
import tensorflow as tf
from typing import List


class TripletLossDataset:

    @property
    def cluster_labels(self) -> List:
        if self._cluster_labels is None:
            self._cluster_labels = list(range(self.num_clusters))
        return self._cluster_labels

    @property
    def num_examples_per_merchant(self) -> int:
        return self.num_examples // self.num_clusters

    def __init__(self, num_clusters: int, num_examples: int, batch_size: int):
        self.num_clusters = num_clusters
        self.num_examples = num_examples
        self.batch_size = batch_size
        self._cluster_labels = None
        self.examples = []
        self.labels = []
        self.dataset = self.build_dataset()

    def build_dataset(self) -> tf.data.Dataset:
        for i, m in enumerate(self.cluster_labels):
            cluster_feat = np.zeros(self.num_clusters)
            cluster_feat[i] = 1.0
            a_feat = np.zeros(4)
            a_feat[random.choice(list(range(4)))] = 1.0
            b_feat = np.zeros(4)
            b_feat[random.choice(list(range(4)))] = 1.0
            for t in range(self.num_examples_per_merchant):
                random_feat = np.zeros(4)
                random_feat[random.choice(list(range(4)))] = 1.0
                self.examples.append(np.concatenate([cluster_feat, a_feat, b_feat, random_feat]))
                self.labels.append(float(m))

        dataset = tf.data.Dataset.from_tensor_slices((self.examples, self.labels))
        dataset = dataset.batch(self.batch_size).shuffle(1024).prefetch(self.batch_size)
        return dataset
