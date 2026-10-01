"""
Visualization Module for Multi-Object Tracking
Project: AIML-03 - Multi-Object Tracking Under Occlusion

Features:
- Distinct, deterministic color per Track ID
- Clean bounding box and ID label rendering
- Trajectory trails showing recent motion history
- Standalone trajectory plot generation for reports and analysis
"""

import os
from collections import defaultdict, deque
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Handles rendering of tracked entities, trajectories, and statistics."""

    # Pre-defined vibrant distinct colors (BGR format for OpenCV)
    COLOR_PALETTE = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Lime Green
        (255, 128, 0),    # Bright Blue
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (180, 105, 255),  # Hot Pink
        (50, 205, 50),    # Lime
        (238, 130, 238),  # Violet
        (0, 215, 255),    # Gold
        (255, 191, 0),    # Deep Sky Blue
        (127, 255, 0),    # Chartreuse
    ]

    def __init__(self, trail_length: int = 30, bbox_thickness: int = 2):
        """
        Args:
            trail_length: Number of past center-point positions to retain in trajectory trail
            bbox_thickness: Thickness in pixels of drawn bounding boxes
        """
        self.trail_length = trail_length
        self.bbox_thickness = bbox_thickness
        # Map: track_id -> deque of (center_x, center_y) points
        self.trajectories = defaultdict(lambda: deque(maxlen=self.trail_length))
        # Complete history for final analysis plots: track_id -> list of (frame, cx, cy)
        self.full_history = defaultdict(list)

    def get_color(self, track_id: int):
        """Deterministically selects a color based on track ID."""
        return self.COLOR_PALETTE[int(track_id) % len(self.COLOR_PALETTE)]

    def draw_tracks(self, frame: np.ndarray, tracked_objects: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """
        Renders bounding boxes, IDs, and trajectory trails onto the video frame.

        Args:
            frame: Input video frame (BGR)
            tracked_objects: Array of shape (M, 5) -> [[x1, y1, x2, y2, id], ...]
            frame_idx: Current frame sequence index

        Returns:
            annotated_frame: Frame with drawn overlays
        """
        annotated = frame.copy()

        # Update trajectories and draw trails
        for obj in tracked_objects:
            x1, y1, x2, y2, tid = obj
            track_id = int(tid)
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            # Record point
            self.trajectories[track_id].append((cx, cy))
            self.full_history[track_id].append((frame_idx, cx, cy))

            color = self.get_color(track_id)

            # Draw trajectory trail with fading line thickness
            trail_pts = list(self.trajectories[track_id])
            for i in range(1, len(trail_pts)):
                thickness = int(np.sqrt(float(i) / len(trail_pts)) * self.bbox_thickness * 1.5) + 1
                cv2.line(annotated, trail_pts[i - 1], trail_pts[i], color, thickness, lineType=cv2.LINE_AA)

            # Draw center point
            cv2.circle(annotated, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Draw bounding box
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, self.bbox_thickness)

            # Draw ID label tag
            label = f"ID: {track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Label banner
            tag_top = max(0, iy1 - text_h - 8)
            cv2.rectangle(annotated, (ix1, tag_top), (ix1 + text_w + 8, iy1), color, -1)
            cv2.putText(
                annotated,
                label,
                (ix1 + 4, iy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),  # High-contrast black text
                1,
                lineType=cv2.LINE_AA
            )

        # Draw Header HUD
        hud_text = f"Frame: {frame_idx:04d} | Active Tracks: {len(tracked_objects)}"
        cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def save_trajectory_plot(self, output_path: str, image_size: tuple = (800, 600)):
        """
        Saves a 2D trajectory bird's-eye/spatial plot of all tracked identities.

        Args:
            output_path: Path where PNG figure is saved
            image_size: (width, height)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#1e1e1e")

        for track_id, points in self.full_history.items():
            if len(points) < 2:
                continue
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]

            # Invert Y to match image coordinates (origin at top-left)
            ax.plot(xs, ys, marker="o", markersize=3, label=f"Person ID {track_id}", linewidth=2)
            # Annotate start and end
            ax.text(xs[0], ys[0], f" Start {track_id}", color="white", fontsize=8)
            ax.text(xs[-1], ys[-1], f" End {track_id}", color="white", fontsize=8)

        ax.invert_yaxis()
        ax.set_title("Customer Trajectories (Ceiling View)", color="white", fontsize=14, weight="bold")
        ax.set_xlabel("Horizontal Position (X px)", color="white")
        ax.set_ylabel("Vertical Position (Y px)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#555555")

        ax.legend(facecolor="#2b2b2b", edgecolor="white", labelcolor="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Visualizer] Trajectory figure saved to: {output_path}")


if __name__ == "__main__":
    print("--- Testing Visualizer Module ---")
    vis = Visualizer(trail_length=10)

    # Create dummy frame
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)

    # Simulate 5 steps of 2 people moving
    for step in range(5):
        tracks = np.array([
            [50 + step * 20, 100, 100 + step * 20, 220, 1],
            [400 - step * 20, 150, 450 - step * 20, 270, 2],
        ])
        canvas = vis.draw_tracks(canvas, tracks, frame_idx=step)

    test_out = os.path.join("results", "figures", "visualizer_test.jpg")
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    cv2.imwrite(test_out, canvas)
    print(f"Annotated test frame saved to: {test_out}")

    plot_out = os.path.join("results", "figures", "trajectories_overview.png")
    vis.save_trajectory_plot(plot_out)
    print("Visualizer tests passed successfully!")
