"""
Multi-Object Tracking Evaluation Module (CLEAR MOT Metrics)
Project: AIML-03 - Multi-Object Tracking Under Occlusion

Standard Evaluation Metrics Implemented:
1. MOTA (Multiple Object Tracking Accuracy):
   MOTA = 1 - (FN + FP + IDSW) / Total_GT
   Measures overall tracking fidelity combining false negatives (misses),
   false positives (spurious tracks), and identity switches.

2. IDSW (Identity Switches):
   Number of times a tracked ground-truth target is assigned a new/different identity.
   Critical indicator of how well the tracker handles occlusion and trajectory crossing.

3. FN (False Negatives / Misses):
   Ground truth pedestrians that were not detected or matched by the tracker.

4. FP (False Positives):
   Detections/tracks output by the system that do not correspond to any real person.

5. MOTP (Multiple Object Tracking Precision):
   Average overlap (IoU) of matched bounding boxes with ground truth.

6. IDF1 (Identification F1 Score):
   Harmonic mean of identification precision and identification recall.
"""

import os
import sys
import glob
import json
import argparse
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.yolo_detector import YOLODetector
from src.tracker.sort import Sort
from src.tracker.association import iou_batch


def parse_gt_file(gt_path: str, min_visibility: float = 0.1) -> dict:
    """
    Parses MOT format gt.txt.
    Line format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    Returns: dict mapping frame_idx (int) -> list of [x1, y1, x2, y2, gt_id]
    """
    gt_by_frame = {}
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            obj_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            x2 = x1 + w
            y2 = y1 + h

            # Check class if available (class 1 is person in MOT)
            if len(parts) >= 8:
                cls_id = int(parts[7])
                if cls_id != 1:
                    continue

            # Check visibility if available (filter completely occluded targets if requested)
            if len(parts) >= 9:
                vis = float(parts[8])
                if vis < min_visibility:
                    continue

            if frame_idx not in gt_by_frame:
                gt_by_frame[frame_idx] = []
            gt_by_frame[frame_idx].append([x1, y1, x2, y2, obj_id])

    return gt_by_frame


def evaluate_tracker_on_sequence(seq_dir: str,
                                 detector: YOLODetector,
                                 tracker: Sort,
                                 iou_eval_threshold: float = 0.5,
                                 min_visibility: float = 0.1) -> dict:
    """
    Runs tracking on a sequence and evaluates CLEAR MOT metrics against gt.txt.
    """
    img_dir = os.path.join(seq_dir, "img1")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")

    gt_by_frame = parse_gt_file(gt_path, min_visibility=min_visibility)
    frame_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                         glob.glob(os.path.join(img_dir, "*.png")))

    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {img_dir}")

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_iou = 0.0

    # Tracking identity memory: gt_id -> last assigned tracker_id
    gt_to_trk_assignment = {}

    for frame_idx, img_path in enumerate(frame_files, start=1):
        frame = cv2.imread(img_path)
        dets = detector.detect(frame)
        active_tracks = tracker.update(dets)

        # Ground truth for this frame
        current_gt = gt_by_frame.get(frame_idx, [])
        gt_boxes = np.array([g[:4] for g in current_gt], dtype=np.float32) if current_gt else np.empty((0, 4))
        gt_ids = [int(g[4]) for g in current_gt]

        trk_boxes = active_tracks[:, :4] if len(active_tracks) > 0 else np.empty((0, 4))
        trk_ids = [int(t[4]) for t in active_tracks] if len(active_tracks) > 0 else []

        total_gt += len(gt_ids)

        if len(gt_boxes) == 0:
            total_fp += len(trk_boxes)
            continue
        if len(trk_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        # Match GT to Tracks using Hungarian Algorithm on IoU
        iou_matrix = iou_batch(gt_boxes, trk_boxes)
        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_trk = set()

        for g_idx, t_idx in zip(row_ind, col_ind):
            match_iou = float(iou_matrix[g_idx, t_idx])
            if match_iou >= iou_eval_threshold:
                matched_gt.add(g_idx)
                matched_trk.add(t_idx)
                total_tp += 1
                total_iou += match_iou

                gid = gt_ids[g_idx]
                tid = trk_ids[t_idx]

                # Check for ID Switch
                if gid in gt_to_trk_assignment:
                    if gt_to_trk_assignment[gid] != tid:
                        total_id_switches += 1
                        gt_to_trk_assignment[gid] = tid
                else:
                    gt_to_trk_assignment[gid] = tid

        # Misses and False alarms
        fn_count = len(gt_boxes) - len(matched_gt)
        fp_count = len(trk_boxes) - len(matched_trk)
        total_fn += fn_count
        total_fp += fp_count

    # Calculate metrics
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / float(total_gt) if total_gt > 0 else 0.0
    motp = (total_iou / total_tp) if total_tp > 0 else 0.0
    precision = (total_tp / float(total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / float(total_gt)) if total_gt > 0 else 0.0
    idf1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "sequence": str(os.path.basename(os.path.normpath(seq_dir))),
        "total_frames": int(len(frame_files)),
        "total_ground_truth": int(total_gt),
        "true_positives": int(total_tp),
        "false_positives": int(total_fp),
        "false_negatives": int(total_fn),
        "id_switches": int(total_id_switches),
        "mota": float(round(mota, 4)),
        "mota_percent": float(round(mota * 100.0, 2)),
        "motp": float(round(motp, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "idf1": float(round(idf1, 4))
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SORT Tracking Metrics on MOT Sequence")
    parser.add_argument("--sequence", "-s", type=str, default="data/sample/retail_seq",
                        help="Path to sequence directory containing img1/ and gt/gt.txt")
    parser.add_argument("--output", "-o", type=str, default="results/metrics/evaluation_report.json",
                        help="Path to save evaluation metrics JSON")
    parser.add_argument("--csv", type=str, default="results/metrics/evaluation_report.csv",
                        help="Path to save evaluation metrics CSV")
    parser.add_argument("--min-vis", type=float, default=0.1,
                        help="Minimum visibility threshold to include ground truth target")

    args = parser.parse_args()

    print("==================================================")
    print("      MULTI-OBJECT TRACKING EVALUATION (SORT)     ")
    print("==================================================")
    print(f"Sequence Directory: {args.sequence}")

    detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.35)
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

    metrics = evaluate_tracker_on_sequence(args.sequence, detector, tracker, min_visibility=args.min_vis)

    # Display Metrics Table
    print("\n---------------- EVALUATION RESULTS ----------------")
    print(f"Total Frames Processed: {metrics['total_frames']}")
    print(f"Ground Truth Targets:   {metrics['total_ground_truth']}")
    print(f"True Positives (TP):    {metrics['true_positives']}")
    print(f"False Negatives (FN):   {metrics['false_negatives']} (Missed detections)")
    print(f"False Positives (FP):   {metrics['false_positives']} (Spurious boxes)")
    print(f"ID Switches (IDSW):     {metrics['id_switches']}")
    print("----------------------------------------------------")
    print(f"MOTA:                   {metrics['mota_percent']}% ({metrics['mota']})")
    print(f"MOTP:                   {metrics['motp']:.4f} (Avg Box IoU Overlap)")
    print(f"Precision:              {metrics['precision']:.4f}")
    print(f"Recall:                 {metrics['recall']:.4f}")
    print(f"IDF1 Score:             {metrics['idf1']:.4f}")
    print("----------------------------------------------------\n")

    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f_json:
        json.dump(metrics, f_json, indent=4)
    print(f"[Export] Saved JSON metrics to: {args.output}")

    # Save to CSV
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(args.csv, index=False)
    print(f"[Export] Saved CSV metrics to:  {args.csv}")


if __name__ == "__main__":
    main()
