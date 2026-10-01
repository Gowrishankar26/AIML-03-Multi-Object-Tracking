"""
SORT: Simple Online and Realtime Tracking with Occlusion Handling
Project: AIML-03 - Multi-Object Tracking Under Occlusion

In Simple Terms:
- SORT combines YOLO detections, Kalman Filter predictions, and Hungarian matching.
- It assigns a persistent integer ID (ID 1, ID 2, ...) to every person.
- Occlusion Handling:
  When a person is temporarily hidden (behind a pillar or another shopper),
  their track is NOT deleted immediately. Instead, `max_age` allows the Kalman
  filter to maintain their identity and project their trajectory. When they
  reappear within `max_age` frames, the Hungarian algorithm matches them back
  to their original ID rather than creating a new false identity!
"""

import numpy as np
from .kalman_filter import KalmanBoxTrackerFilter
from .association import associate_detections_to_trackers


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked person instances.
    """
    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initializes a tracker using an initial bounding box [x1, y1, x2, y2].
        """
        self.kf = KalmanBoxTrackerFilter(bbox[:4])
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox: np.ndarray):
        """
        Updates the state estimate with an observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox[:4])

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # If the track has not been updated recently, hit streak breaks
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.age += 1
        pos = self.kf.predict()
        self.history.append(pos)
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.
        """
        return self.kf.get_state()


class Sort:
    """
    SORT tracker managing multiple person tracks across video frames.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Maximum frames to retain a track without detections (occlusion tolerance)
            min_hits: Minimum confirmed detections before a track is reported
            iou_threshold: Minimum IoU overlap for Hungarian matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        # Reset track counter for fresh tracking runs
        KalmanBoxTracker.count = 0

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """
        Process detections for the current frame and return active tracks.

        Args:
            dets: np.ndarray of shape (N, 5) -> [[x1, y1, x2, y2, score], ...]

        Returns:
            np.ndarray: Tracked boxes of shape (M, 5) -> [[x1, y1, x2, y2, id], ...]
        """
        self.frame_count += 1

        # Step 1: Predict new locations of existing tracks using Kalman Filter
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # Flag invalid bounding boxes (NaN or infinite coordinates)
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Step 2: Associate detections to predicted tracks via Hungarian Algorithm
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks[:, :4], self.iou_threshold
        )

        # Step 3: Update matched trackers with corresponding detection
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Step 4: Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Step 5: Filter output and remove dead tracks exceeding max_age
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # Track is reported if it was updated recently and has sufficient hit confirmation
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # Remove tracks that have been unseen for longer than max_age
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


if __name__ == "__main__":
    print("--- Running Phase 7 & Phase 8: SORT Occlusion Verification Test ---")
    tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)

    # Scenario:
    # Person A starts at X=100 and moves right by 10 px/frame
    # Person B starts at X=500 and moves left by 10 px/frame
    # At frame 5, Person A is temporarily OCCLUDED for 4 frames (frames 5, 6, 7, 8)
    # At frame 9, Person A reappears at X=190.
    # We verify that Person A still has the SAME ID upon reappearance!

    recorded_ids_A = []
    recorded_ids_B = []

    for frame in range(1, 14):
        pos_A = [100 + (frame - 1) * 10, 100, 150 + (frame - 1) * 10, 250, 0.95]
        pos_B = [500 - (frame - 1) * 10, 100, 550 - (frame - 1) * 10, 250, 0.95]

        # Simulate occlusion for Person A on frames 5, 6, 7, 8
        if 5 <= frame <= 8:
            current_dets = np.array([pos_B], dtype=np.float32)
            occlusion_status = "Person A OCCLUDED"
        else:
            current_dets = np.array([pos_A, pos_B], dtype=np.float32)
            occlusion_status = "Both visible"

        active_tracks = tracker.update(current_dets)

        # Find which track IDs were assigned
        frame_ids = []
        for trk in active_tracks:
            x1, y1, x2, y2, tid = trk
            frame_ids.append(int(tid))
            # Determine if this box is near Person A or B
            center_x = (x1 + x2) / 2.0
            if center_x < 350:
                recorded_ids_A.append((frame, int(tid)))
            else:
                recorded_ids_B.append((frame, int(tid)))

        print(f"Frame {frame:2d} ({occlusion_status:18s}) -> Active Track IDs: {frame_ids}")

    # Check ID stability
    unique_ids_A = set([tid for _, tid in recorded_ids_A])
    unique_ids_B = set([tid for _, tid in recorded_ids_B])

    print(f"\nPerson A Unique Assigned IDs across entire run: {unique_ids_A}")
    print(f"Person B Unique Assigned IDs across entire run: {unique_ids_B}")

    assert len(unique_ids_A) == 1, f"Occlusion failure: Person A changed ID! Got {unique_ids_A}"
    assert len(unique_ids_B) == 1, f"Person B changed ID! Got {unique_ids_B}"
    print("\nSUCCESS: Person A maintained persistent identity through 4 frames of complete occlusion!")
