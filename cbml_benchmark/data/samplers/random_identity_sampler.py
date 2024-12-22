import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly samples `N` identities (labels), and for each identity,
    randomly samples `K` instances. Therefore, the batch size is `N * K`.

    Args:
    - dataset (BaseDataSet): The dataset containing data and labels.
      It must have a `label_index_dict` that maps labels to indices.
    - num_instances (int): The number of instances per identity in a batch (K).
    - batch_size (int): The total number of samples in a batch (N * K).
    - max_iters (int): The number of batches (iterations) to generate.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        """
        Initializes the sampler.

        Args:
        - dataset: Dataset with `label_index_dict` mapping labels to indices.
        - batch_size: Total number of samples in each batch.
        - num_instances: Number of instances per label in a batch (K).
        - max_iters: Number of iterations (batches to yield).
        """
        self.label_index_dict = dataset.label_index_dict  # Map of label to indices in the dataset.
        self.batch_size = batch_size  # Total samples in a batch (N * K).
        self.K = num_instances  # Number of instances per identity (K).
        self.num_labels_per_batch = self.batch_size // self.K  # Number of unique labels per batch (N).
        self.max_iters = max_iters  # Total number of iterations (batches).
        self.labels = list(self.label_index_dict.keys())  # List of all unique labels in the dataset.

    def __len__(self):
        """
        Returns the total number of iterations (batches) the sampler will generate.
        """
        return self.max_iters

    def __repr__(self):
        """
        Returns a string representation of the sampler for debugging or logging.
        """
        return self.__str__()

    def __str__(self):
        """
        String representation of the sampler showing key parameters.
        """
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def _prepare_batch(self):
        """
        Prepares a dictionary of indices grouped by label for batch sampling.

        Returns:
        - batch_idxs_dict: A dictionary where keys are labels and values are lists of indices
          grouped into chunks of size `K` (num_instances).
        - avai_labels: A list of labels that are available for sampling.
        """
        batch_idxs_dict = defaultdict(list)  # Dictionary to store batches of indices for each label.

        for label in self.labels:
            # Get all indices for the current label.
            idxs = copy.deepcopy(self.label_index_dict[label])
            
            # If there are fewer than `K` indices, oversample to meet the required count.
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            
            # Shuffle indices to ensure randomness.
            random.shuffle(idxs)

            # Split indices into groups of size `K` and store them in the dictionary.
            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        # Make a copy of all labels to track which ones are available for sampling.
        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        """
        Iterates through the sampler to yield batches of indices.

        Yields:
        - A list of indices for each batch, where the batch size is `N * K`.
        """
        # Prepare the initial batches and available labels.
        batch_idxs_dict, avai_labels = self._prepare_batch()

        for _ in range(self.max_iters):
            batch = []  # List to store indices for the current batch.

            # If there are not enough available labels to create a batch, reinitialize batches and labels.
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            # Randomly select `N` labels for the batch.
            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)

            # For each selected label, add `K` instances to the batch.
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)  # Get the next group of `K` indices.
                batch.extend(batch_idxs)  # Add these indices to the batch.
                
                # If all groups for this label are used, remove the label from available labels.
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

            # Yield the current batch of indices.
            yield batch
