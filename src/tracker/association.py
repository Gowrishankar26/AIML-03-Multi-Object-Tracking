"""
Data Association Module: IoU & Hungarian Algorithm
Project: AIML-03 - Multi-Object Tracking Under Occlusion

In Simple Terms:
- IoU (Intersection over Union): Measures how much two bounding boxes overlap.
      IoU = Intersection Area / Union Area
      Values range from 0.0 (no overlap) to 1.0 (exact match).
- Hungarian Algorithm: Finds the optimal global assignment between existing
  person tracks and new detections so that overall overlap is maximized.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        float: IoU value in range [0.0, 1.0]

    Formula:
        Intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *
                       max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        Union = Area(box1) + Area(box2) - Intersection
        IoU = Intersection / Union
    """
    # Coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If there is no overlap, intersection is 0
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Individual box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between N test boxes and M ground-truth/track boxes.

    Args:
        bb_test: np.ndarray of shape (N, 4) -> [[x1, y1, x2, y2], ...]
        bb_gt: np.ndarray of shape (M, 4) -> [[x1, y1, x2, y2], ...]

    Returns:
        np.ndarray: IoU matrix of shape (N, M)
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, 0)      # (1, M, 4)

    # Intersection top-left and bottom-right
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    # Area of each box
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - wh
    union = np.maximum(union, 1e-6)

    return (wh / union).astype(np.float32)


def associate_detections_to_trackers(detections: np.ndarray,
                                     trackers: np.ndarray,
                                     iou_threshold: float = 0.3):
    """
    Assigns detections to tracked persons using the Hungarian Algorithm on IoU cost.

    Args:
        detections: Array of detections, shape (N, 5) -> [[x1, y1, x2, y2, conf], ...]
                    or (N, 4) -> [[x1, y1, x2, y2], ...]
        trackers: Array of predicted track positions, shape (M, 4) -> [[x1, y1, x2, y2], ...]
        iou_threshold: Minimum IoU overlap required to consider a match valid

    Returns:
        matches: Array of matched pairs (N_matches, 2) -> [[det_idx, track_idx], ...]
        unmatched_detections: 1D array of indices of detections with no match
        unmatched_trackers: 1D array of indices of tracks with no match
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    # Strip confidence score if present
    det_boxes = detections[:, :4] if detections.shape[1] >= 4 else detections

    # Calculate IoU matrix of shape (num_detections, num_trackers)
    iou_matrix = iou_batch(det_boxes, trackers)

    # Hungarian algorithm solves minimum weight bipartite matching.
    # To maximize IoU, cost = -IoU matrix.
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_indices = []
    unmatched_detections = set(range(len(detections)))
    unmatched_trackers = set(range(len(trackers)))

    for d, t in zip(row_ind, col_ind):
        # Reject assignments where overlap is less than iou_threshold
        if iou_matrix[d, t] < iou_threshold:
            continue
        matched_indices.append([d, t])
        unmatched_detections.discard(d)
        unmatched_trackers.discard(t)

    if len(matched_indices) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matched_indices, dtype=int)

    return matches, np.array(sorted(unmatched_detections), dtype=int), np.array(sorted(unmatched_trackers), dtype=int)


if __name__ == "__main__":
    print("--- Testing Phase 4: IoU Calculation ---")
    # Test case 1: Identical boxes -> IoU should be 1.0
    box_a = np.array([10, 10, 50, 50])
    box_b = np.array([10, 10, 50, 50])
    res_1 = iou(box_a, box_b)
    print(f"Test 1 (Identical): Expected 1.000, Got: {res_1:.4f}")
    assert np.isclose(res_1, 1.0), "Test 1 failed!"

    # Test case 2: Completely disjoint boxes -> IoU should be 0.0
    box_c = np.array([100, 100, 150, 150])
    res_2 = iou(box_a, box_c)
    print(f"Test 2 (Disjoint):  Expected 0.000, Got: {res_2:.4f}")
    assert np.isclose(res_2, 0.0), "Test 2 failed!"

    # Test case 3: 50% overlap geometry
    # Box A: [0, 0, 10, 10] -> Area = 100
    # Box D: [0, 0, 10, 5]  -> Area = 50 (completely inside bottom half)
    # Intersection = 50, Union = 100 -> IoU = 0.50
    box_d = np.array([0, 0, 10, 5])
    box_e = np.array([0, 0, 10, 10])
    res_3 = iou(box_d, box_e)
    print(f"Test 3 (50% Overlap): Expected 0.500, Got: {res_3:.4f}")
    assert np.isclose(res_3, 0.5), "Test 3 failed!"

    print("\n--- Testing Phase 6: Hungarian Association ---")
    detections = np.array([
        [10, 10, 50, 50, 0.95],    # Matches Tracker 1
        [102, 102, 148, 148, 0.88], # Matches Tracker 2
        [300, 300, 350, 350, 0.80]  # New person (Unmatched detection)
    ])
    trackers = np.array([
        [12, 12, 52, 52],          # Tracker 1 (close to Det 0)
        [100, 100, 150, 150],      # Tracker 2 (close to Det 1)
        [500, 500, 550, 550]       # Tracker 3 (Occluded / departed person)
    ])

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)
    print(f"Matches [Det Index -> Track Index]:\n{matches}")
    print(f"Unmatched Detections (New people): {unmatched_dets}")
    print(f"Unmatched Trackers (Occluded people): {unmatched_trks}")

    assert len(matches) == 2, "Expected 2 matches!"
    assert list(unmatched_dets) == [2], "Det 2 should be unmatched!"
    assert list(unmatched_trks) == [2], "Track 2 should be unmatched!"
    print("\nAll IoU & Hungarian Association tests passed successfully!")
