"""
DeepSORT: Deep Simple Online and Realtime Tracking with Appearance Embeddings
Project: AIML-03 - Multi-Object Tracking Under Occlusion (Bonus Component)

Enhances SORT by incorporating deep appearance feature embeddings.
When two people cross paths or one emerges from occlusion, visual feature similarity
(cosine distance) resolves ambiguity that geometric IoU alone cannot solve.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from scipy.optimize import linear_sum_assignment

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tracker.kalman_filter import KalmanBoxTrackerFilter
from src.tracker.association import iou_batch


class AppearanceFeatureExtractor:
    """Extracts normalized L2 visual feature embeddings from cropped person bounding boxes."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # Use a lightweight MobileNetV3 feature extractor
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        # Remove final classification layer to extract 576-dimensional embedding
        self.extractor = model.features
        self.avgpool = model.avgpool
        self.extractor.eval()
        self.extractor.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),  # Standard pedestrian Re-ID aspect ratio
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extracts appearance embeddings for all bounding boxes in the frame.

        Args:
            frame: Input image (BGR)
            bboxes: (N, 4) array -> [[x1, y1, x2, y2], ...]

        Returns:
            np.ndarray: (N, 576) normalized feature matrix
        """
        if len(bboxes) == 0 or frame is None:
            return np.empty((0, 576), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []
        for box in bboxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))

            # Handle degenerate boxes
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb_crop)
            crops.append(tensor)

        batch_tensors = torch.stack(crops).to(self.device)
        feats = self.avgpool(self.extractor(batch_tensors))
        feats = torch.flatten(feats, 1)

        # L2 Normalization so cosine distance = 1 - dot product
        norms = torch.norm(feats, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-12)
        normalized = feats / norms

        return normalized.cpu().numpy()


class DeepKalmanBoxTracker:
    """Track representation maintaining Kalman filter state and appearance feature gallery."""
    count = 0

    def __init__(self, bbox: np.ndarray, feature: np.ndarray = None, gallery_size: int = 15):
        self.kf = KalmanBoxTrackerFilter(bbox[:4])
        self.time_since_update = 0
        DeepKalmanBoxTracker.count += 1
        self.id = DeepKalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.gallery_size = gallery_size
        self.features = [feature] if feature is not None else []

    def update(self, bbox: np.ndarray, feature: np.ndarray = None):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox[:4])
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.gallery_size:
                self.features.pop(0)

    def predict(self) -> np.ndarray:
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        return self.kf.predict()

    def get_state(self) -> np.ndarray:
        return self.kf.get_state()


class DeepSort:
    """DeepSORT Tracker combining Kalman prediction, IoU distance, and appearance embeddings."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, max_cosine_distance: float = 0.4,
                 lambda_weight: float = 0.4, device: str = "cpu"):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.lambda_weight = lambda_weight  # Weight for IoU vs Appearance in cost matrix
        self.trackers = []
        self.frame_count = 0
        self.feature_extractor = AppearanceFeatureExtractor(device=device)
        DeepKalmanBoxTracker.count = 0

    def update(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        # Extract features for all detections
        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))
        det_features = self.feature_extractor.extract(frame, det_boxes) if len(det_boxes) > 0 else np.empty((0, 576))

        # 1. Predict all tracks
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Association using combined IoU and Appearance Cost
        matched, unmatched_dets, unmatched_trks = self._associate(det_boxes, det_features, trks[:, :4])

        # 3. Update matched trackers
        for m in matched:
            d_idx, t_idx = m
            feat = det_features[d_idx] if len(det_features) > d_idx else None
            self.trackers[t_idx].update(dets[d_idx, :], feat)

        # 4. Create new trackers
        for i in unmatched_dets:
            feat = det_features[i] if len(det_features) > i else None
            trk = DeepKalmanBoxTracker(dets[i, :], feat)
            self.trackers.append(trk)

        # 5. Output confirmed tracks and prune dead tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _associate(self, det_boxes, det_features, trk_boxes):
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(det_boxes)), np.empty((0,), dtype=int)
        if len(det_boxes) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(self.trackers))

        # 1. IoU distance matrix: 1 - IoU
        iou_matrix = iou_batch(det_boxes, trk_boxes)
        iou_dist = 1.0 - iou_matrix

        # 2. Appearance cosine distance matrix
        app_dist = np.ones((len(det_boxes), len(self.trackers)), dtype=np.float32)
        for t_idx, trk in enumerate(self.trackers):
            if len(trk.features) > 0:
                trk_gallery = np.array(trk.features)  # (K, 576)
                # Compute pairwise dot products
                dot_prods = np.dot(det_features, trk_gallery.T)  # (N_dets, K)
                # Cosine distance = 1 - max(dot_product)
                app_dist[:, t_idx] = 1.0 - np.max(dot_prods, axis=1)

        # Combined Cost Matrix
        cost_matrix = self.lambda_weight * iou_dist + (1.0 - self.lambda_weight) * app_dist

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_dets = set(range(len(det_boxes)))
        unmatched_trks = set(range(len(self.trackers)))

        for d, t in zip(row_ind, col_ind):
            # Gating: must satisfy both IoU overlap and appearance distance thresholds
            if iou_matrix[d, t] < self.iou_threshold and app_dist[d, t] > self.max_cosine_distance:
                continue
            matched_indices.append([d, t])
            unmatched_dets.discard(d)
            unmatched_trks.discard(t)

        matches = np.array(matched_indices, dtype=int) if matched_indices else np.empty((0, 2), dtype=int)
        return matches, np.array(sorted(unmatched_dets), dtype=int), np.array(sorted(unmatched_trks), dtype=int)
