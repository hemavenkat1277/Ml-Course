import numpy as np
import pandas as pd
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(X_column, y, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
        return 0
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    n = len(y)
    n_left, n_right = len(y[left_indices]), len(y[right_indices])
    weighted_avg_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    return entropy(y) - weighted_avg_entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or num_labels == 1 or num_samples == 0):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_idx in range(num_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold
        
        left_indices = X[:, split_idx] <= split_thresh
        right_indices = X[:, split_idx] > split_thresh
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(split_idx, split_thresh, left, right)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Example usage
if __name__ == "__main__":
    # Sample data (X: features, y: labels)
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1, 1, 0])
    
    # Train decision tree
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    
    # Predictions
    predictions = clf.predict(X)
    print("Predictions:", predictions)
