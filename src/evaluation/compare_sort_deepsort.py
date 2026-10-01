"""
Benchmarking Comparison: SORT vs. DeepSORT
Project: AIML-03 - Multi-Object Tracking Under Occlusion (Bonus Evaluation)

Directly compares SORT (geometry-only) against DeepSORT (geometry + appearance embeddings)
under identical video sequence conditions.
"""

import os
import sys
import json
import glob
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
from src.tracker.deepsort import DeepSort
from src.tracker.association import iou_batch
from src.evaluation.evaluate import parse_gt_file


def run_benchmark(seq_dir: str = "data/sample/retail_seq",
                  output_json: str = "results/metrics/sort_vs_deepsort.json"):
    print("==========================================================")
    print("         BENCHMARK COMPARISON: SORT vs. DeepSORT          ")
    print("==========================================================")

    img_dir = os.path.join(seq_dir, "img1")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")

    gt_by_frame = parse_gt_file(gt_path, min_visibility=0.1)
    frame_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                         glob.glob(os.path.join(img_dir, "*.png")))

    detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.35)

    trackers = {
        "SORT": Sort(max_age=30, min_hits=2, iou_threshold=0.3),
        "DeepSORT": DeepSort(max_age=30, min_hits=2, iou_threshold=0.3, max_cosine_distance=0.45)
    }

    comparison_results = {}

    for name, trk in trackers.items():
        print(f"\n[Evaluating {name}]...")
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_iou = 0.0
        gt_to_trk = {}

        for frame_idx, img_path in enumerate(frame_files, start=1):
            frame = cv2.imread(img_path)
            dets = detector.detect(frame)

            if name == "DeepSORT":
                active_tracks = trk.update(frame, dets)
            else:
                active_tracks = trk.update(dets)

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

            iou_matrix = iou_batch(gt_boxes, trk_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched_gt = set()
            matched_trk = set()

            for g_idx, t_idx in zip(row_ind, col_ind):
                match_iou = float(iou_matrix[g_idx, t_idx])
                if match_iou >= 0.5:
                    matched_gt.add(g_idx)
                    matched_trk.add(t_idx)
                    total_tp += 1
                    total_iou += match_iou

                    gid = gt_ids[g_idx]
                    tid = trk_ids[t_idx]

                    if gid in gt_to_trk:
                        if gt_to_trk[gid] != tid:
                            total_idsw += 1
                            gt_to_trk[gid] = tid
                    else:
                        gt_to_trk[gid] = tid

            total_fn += (len(gt_boxes) - len(matched_gt))
            total_fp += (len(trk_boxes) - len(matched_trk))

        mota = 1.0 - (total_fn + total_fp + total_idsw) / float(total_gt) if total_gt > 0 else 0.0
        motp = (total_iou / total_tp) if total_tp > 0 else 0.0
        prec = (total_tp / float(total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        rec = (total_tp / float(total_gt)) if total_gt > 0 else 0.0

        comparison_results[name] = {
            "MOTA": f"{mota * 100.0:.2f}%",
            "MOTA_raw": round(mota, 4),
            "ID_Switches": total_idsw,
            "MOTP": round(motp, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn
        }

    # Print requested Comparison Table
    print("\n==========================================================")
    print("                 EXPERIMENTAL COMPARISON                  ")
    print("==========================================================")
    print(f"{'Metric':<20} {'SORT':<18} {'DeepSORT':<18}")
    print("----------------------------------------------------------")
    print(f"{'MOTA':<20} {comparison_results['SORT']['MOTA']:<18} {comparison_results['DeepSORT']['MOTA']:<18}")
    print(f"{'ID Switches':<20} {comparison_results['SORT']['ID_Switches']:<18} {comparison_results['DeepSORT']['ID_Switches']:<18}")
    print(f"{'MOTP (Overlap)':<20} {comparison_results['SORT']['MOTP']:<18} {comparison_results['DeepSORT']['MOTP']:<18}")
    print(f"{'Precision':<20} {comparison_results['SORT']['Precision']:<18} {comparison_results['DeepSORT']['Precision']:<18}")
    print(f"{'Recall':<20} {comparison_results['SORT']['Recall']:<18} {comparison_results['DeepSORT']['Recall']:<18}")
    print("==========================================================")

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f_out:
        json.dump(comparison_results, f_out, indent=4)
    print(f"\n[Export] Comparison results saved to: {output_json}")


if __name__ == "__main__":
    run_benchmark()
