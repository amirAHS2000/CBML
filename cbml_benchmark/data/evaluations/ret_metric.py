import numpy as np

class RetMetric(object):
    def __init__(self, feats, labels):
        """
        Initializes the retrieval metric class.

        Parameters:
        feats: array-like or list
            - If a list: [gallery_feats, query_feats], where each is a 2D array of features.
            - If not a list: a single 2D array where gallery and query features are the same.
        labels: array-like or list
            - If a list: [gallery_labels, query_labels], corresponding to gallery and query features.
            - If not a list: a single array where gallery and query labels are the same.
        """
        
        # If `feats` is a list, we assume gallery and query are distinct
        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False  # Flag indicating gallery and query are distinct

            # Separate gallery and query features and labels
            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            # If not a list, assume gallery and query are identical
            self.is_equal_query = True  # Flag indicating gallery and query are identical
            
            # Use the same features and labels for both gallery and query
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        # Compute the similarity matrix using dot product
        # Each element sim_mat[i][j] is the similarity between query[i] and gallery[j]
        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        """
        Computes the recall at rank `k`.

        Parameters:
        k: int
            The threshold for the maximum number of negative samples allowed 
            above the positive similarity threshold.

        Returns:
        float
            Recall score as the fraction of queries for which the positive matches
            are ranked within the top-k.
        """
        
        # Total number of queries
        m = len(self.sim_mat)

        # Counter for successful matches
        match_counter = 0

        # Iterate over each query
        for i in range(m):
            # Extract similarities for positive matches (labels match)
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            
            # Extract similarities for negative matches (labels do not match)
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            # Determine the similarity threshold
            # - For self-retrieval (is_equal_query), use the second-highest positive similarity
            #   to avoid matching with itself.
            # - Otherwise, use the maximum positive similarity.
            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            """
            The below line is checking if the number of irrelevant items (those with different labels)
            in the top k ranked by similarity is less than k. If so, it counts the query
            as a success, meaning that relevant items have a chance of appearing in the top k,
            which is the basis for computing recall at k.
            """
            # Check if fewer than `k` negatives exceed the positive similarity threshold
            if np.sum(neg_sim > thresh) < k:
                match_counter += 1  # Count as a successful match

        # Compute recall as the fraction of successful matches
        return float(match_counter) / m
