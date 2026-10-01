"""
YOLO Detector Module for Person Detection
Project: AIML-03 - Multi-Object Tracking Under Occlusion

This module wraps Ultralytics YOLO to detect people in video frames.
Only detections belonging to class 0 ('person' in COCO dataset) are retained.

Internal Detection Format:
    Each detected person is represented as:
    [x1, y1, x2, y2, confidence]
    where:
        x1: Leftmost horizontal pixel coordinate (top-left X)
        y1: Topmost vertical pixel coordinate (top-left Y)
        x2: Rightmost horizontal pixel coordinate (bottom-right X)
        y2: Bottommost vertical pixel coordinate (bottom-right Y)
        confidence: Model prediction confidence score between 0.0 and 1.0
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Wrapper around Ultralytics YOLO for person-only detection."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.40, device: str = "cpu"):
        """
        Initialize the YOLO detector.

        Args:
            model_name: Pretrained YOLO model file (e.g., 'yolov8n.pt')
            conf_threshold: Minimum confidence score to accept a detection
            device: Computing device ('cpu' or '0' for GPU)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.target_class_id = 0  # In COCO dataset, 0 is 'person'

        print(f"[YOLODetector] Loading pretrained model: {model_name} on {device}...")
        self.model = YOLO(model_name)
        print(f"[YOLODetector] Model loaded successfully. Filtered for Person (Class ID: {self.target_class_id}).")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run person detection on a single image/frame.

        Args:
            frame: Input image/frame in BGR format (numpy array)

        Returns:
            np.ndarray: Detections of shape (N, 5), where each row is:
                        [x1, y1, x2, y2, confidence]
                        Returns np.empty((0, 5)) if no persons are detected.
        """
        if frame is None or frame.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Run inference (verbose=False to keep logs clean during video processing)
        results = self.model.predict(
            source=frame,
            classes=[self.target_class_id],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Extract coordinates and confidence
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

        if len(detections) == 0:
            return np.empty((0, 5), dtype=np.float32)

        return np.array(detections, dtype=np.float32)

    def draw_detections(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Draw detected person bounding boxes and confidence labels on the frame.

        Args:
            frame: Input frame (BGR)
            detections: Bounding boxes with shape (N, 5) -> [x1, y1, x2, y2, conf]

        Returns:
            annotated_frame: Frame with drawn bounding boxes and text labels
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            color = (0, 255, 0)  # Bright Green for Person
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Format label: "Person: 0.94"
            label = f"Person: {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw filled background for label text for high legibility
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 4, max(0, y1)),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text on green background
                1,
                cv2.LINE_AA
            )

        return annotated_frame


if __name__ == "__main__":
    # Test script for Phase 2 verification
    print("--- Running Phase 2: YOLO Person Detection Test ---")
    detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.35)

    # Use Ultralytics built-in sample or download a standard retail/street test frame
    from ultralytics.utils import ASSETS
    sample_img_path = os.path.join(ASSETS, "bus.jpg")

    if not os.path.exists(sample_img_path):
        # Fallback: create a synthetic test image with a person silhouette or load sample
        print("Ultralytics sample image not found at default path, testing with synthetic image.")
        test_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    else:
        test_frame = cv2.imread(sample_img_path)
        print(f"Loaded test frame from: {sample_img_path} ({test_frame.shape[1]}x{test_frame.shape[0]})")

    detections = detector.detect(test_frame)
    print(f"\nTotal Person Detections found: {len(detections)}")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf = det
        print(f"  Person #{i+1}: Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Confidence={conf:.3f}")

    annotated = detector.draw_detections(test_frame, detections)

    output_dir = os.path.join("results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "detection_sample.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated detection result to: {out_path}")
    print("--- Phase 2 Completed Successfully! ---")
