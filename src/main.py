"""
Multi-Object Tracking Pipeline Main Runner
Project: AIML-03 - Multi-Object Tracking Under Occlusion

Runs the complete end-to-end pipeline:
Video/Sequence -> YOLO Person Detector -> SORT Tracker (Kalman + Hungarian) -> Visualizer -> Annotated Video Output
"""

import os
import sys
import glob
import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.yolo_detector import YOLODetector
from src.tracker.sort import Sort
from src.visualization.visualize import Visualizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration settings from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    print(f"[Warning] Config file not found at {config_path}, using defaults.")
    return {
        "detector": {"model_name": "yolov8n.pt", "conf_threshold": 0.40, "device": "cpu"},
        "tracker": {"max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
        "visualization": {"trail_length": 30, "bbox_thickness": 2}
    }


def generate_occlusion_demo_video(output_video_path: str = "data/sample/occlusion_demo.mp4",
                                  width: int = 800, height: int = 500, num_frames: int = 120):
    """
    Generates a realistic synthetic retail ceiling-view video with two shoppers crossing paths
    and a pillar causing a complete temporary occlusion.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

    print(f"[Demo Generator] Creating realistic occlusion demo video: {output_video_path}...")

    # Obstacle (Pillar in retail store)
    pillar_x1, pillar_y1, pillar_x2, pillar_y2 = 360, 100, 440, 400

    for f in range(num_frames):
        # Store floor background
        frame = np.full((height, width, 3), (230, 230, 235), dtype=np.uint8)

        # Floor tile grid pattern
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (210, 210, 215), 1)
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (210, 210, 215), 1)

        # Draw Person A (Shopper moving Left to Right)
        # Position X goes from 100 to 700
        ax = int(100 + (f / num_frames) * 600)
        ay = 240
        # Draw realistic person representation (torso + head)
        cv2.circle(frame, (ax, ay - 45), 18, (140, 70, 40), -1)  # Head
        cv2.rectangle(frame, (ax - 22, ay - 25), (ax + 22, ay + 65), (200, 110, 60), -1)  # Torso/Legs

        # Draw Person B (Shopper moving Right to Left)
        # Position X goes from 700 down to 100
        bx = int(700 - (f / num_frames) * 600)
        by = 240
        cv2.circle(frame, (bx, by - 45), 18, (40, 120, 180), -1)
        cv2.rectangle(frame, (bx - 22, by - 25), (bx + 22, by + 65), (50, 160, 230), -1)

        # Draw Pillar (Architectural occlusion zone) in foreground
        cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (90, 95, 105), -1)
        cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (40, 45, 50), 3)
        cv2.putText(frame, "RETAIL PILLAR", (pillar_x1 + 8, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Overlay title
        cv2.putText(frame, "Retail Store Ceiling Camera - Occlusion Benchmark",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)

        writer.write(frame)

    writer.release()
    print(f"[Demo Generator] Successfully created: {output_video_path}")
    return output_video_path


def run_pipeline(input_source: str,
                 output_video: str,
                 config: dict,
                 show: bool = False,
                 export_trajectories: bool = True):
    """
    Executes the end-to-end tracking pipeline.
    """
    det_cfg = config.get("detector", {})
    trk_cfg = config.get("tracker", {})
    vis_cfg = config.get("visualization", {})

    # Initialize components
    detector = YOLODetector(
        model_name=det_cfg.get("model_name", "yolov8n.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.40),
        device=det_cfg.get("device", "cpu")
    )

    tracker = Sort(
        max_age=trk_cfg.get("max_age", 30),
        min_hits=trk_cfg.get("min_hits", 3),
        iou_threshold=trk_cfg.get("iou_threshold", 0.3)
    )

    visualizer = Visualizer(
        trail_length=vis_cfg.get("trail_length", 30),
        bbox_thickness=vis_cfg.get("bbox_thickness", 2)
    )

    # Determine input type (Video file, image directory, or camera)
    is_image_dir = os.path.isdir(input_source)
    if is_image_dir:
        frame_files = sorted(glob.glob(os.path.join(input_source, "*.jpg")) +
                             glob.glob(os.path.join(input_source, "*.png")))
        if not frame_files:
            raise FileNotFoundError(f"No .jpg or .png images found in directory: {input_source}")
        sample = cv2.imread(frame_files[0])
        height, width = sample.shape[:2]
        fps = 25.0
        total_frames = len(frame_files)
        cap = None
        print(f"[Pipeline] Processing image sequence directory: {input_source} ({total_frames} frames)")
    else:
        cap = cv2.VideoCapture(int(input_source) if input_source.isdigit() else input_source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {input_source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None
        frame_files = None
        print(f"[Pipeline] Processing video: {input_source} ({width}x{height} @ {fps:.1f} FPS)")

    # Prepare VideoWriter
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Tracking Progress", unit="frame")

    unique_tracked_ids = set()

    try:
        while True:
            if is_image_dir:
                if frame_idx >= len(frame_files):
                    break
                frame = cv2.imread(frame_files[frame_idx])
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            # 1. Detect Persons via YOLO
            detections = detector.detect(frame)

            # 2. Update SORT Tracker
            tracked_objects = tracker.update(detections)

            for trk in tracked_objects:
                unique_tracked_ids.add(int(trk[4]))

            # 3. Visualize Tracks & Trajectories
            annotated_frame = visualizer.draw_tracks(frame, tracked_objects, frame_idx=frame_idx)

            # Write to output video
            out_writer.write(annotated_frame)

            if show:
                cv2.imshow("Multi-Object Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        if cap is not None:
            cap.release()
        out_writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n[Pipeline Complete]")
    print(f"  Frames Processed: {frame_idx}")
    print(f"  Total Unique Identities Tracked: {len(unique_tracked_ids)} ({sorted(list(unique_tracked_ids))})")
    print(f"  Output Tracked Video Saved: {output_video}")

    if export_trajectories:
        fig_path = os.path.join("results", "figures", "video_trajectories.png")
        visualizer.save_trajectory_plot(fig_path, image_size=(width, height))
        print(f"  Trajectory Map Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Object Tracking Under Occlusion (AIML-03)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input video file or image sequence directory")
    parser.add_argument("--output", "-o", type=str, default="results/videos/tracked.mp4",
                        help="Path for output annotated MP4 video")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--demo-occlusion", action="store_true",
                        help="Generate and track a synthetic retail occlusion benchmark sequence")
    parser.add_argument("--show", action="store_true",
                        help="Display video output in OpenCV window during processing")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.demo_occlusion or args.input is None:
        print("[Notice] No input specified or --demo-occlusion requested. Generating occlusion demo sequence.")
        demo_path = "data/sample/retail_occlusion_test.mp4"
        generate_occlusion_demo_video(demo_path)
        args.input = demo_path

    run_pipeline(
        input_source=args.input,
        output_video=args.output,
        config=config,
        show=args.show
    )


if __name__ == "__main__":
    main()
