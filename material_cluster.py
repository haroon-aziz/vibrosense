import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (238, 130, 238),
    (255, 215, 0),
    (0, 206, 209),
]


class MaterialClusterer:
    def __init__(self, n_clusters=4, random_state=0):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)

    def fit_predict(self, feature_matrix):
        scaled = self.scaler.fit_transform(feature_matrix)
        labels = self.model.fit_predict(scaled)
        return labels

    def predict(self, feature_matrix):
        scaled = self.scaler.transform(feature_matrix)
        return self.model.predict(scaled)


def draw_overlay(frame, coords, labels, patch_size=16, alpha=0.45):
    overlay = frame.copy()
    for (y, x), label in zip(coords, labels):
        color = PALETTE[label % len(PALETTE)]
        cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), color, -1)
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return blended
