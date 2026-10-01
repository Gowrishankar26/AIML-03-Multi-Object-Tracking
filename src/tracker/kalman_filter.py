"""
Kalman Filter for Bounding Box Motion Estimation
Project: AIML-03 - Multi-Object Tracking Under Occlusion

In Simple Terms:
- Kalman Filter = Predicts where a person will move next.
- When a person walks, their velocity carries them forward.
- If a person is temporarily hidden (occluded), the detector sees nothing.
  The Kalman filter steps in to PREDICT their position until they reappear!

Math Pipeline:
    1. Prediction (Time Step):
       x_hat = F * x
       P = F * P * F^T + Q

    2. Measurement:
       z = [u, v, s, r]^T  (From YOLO detection)

    3. Update (Correction Step):
       y = z - H * x_hat          (Innovation: measurement residual)
       S = H * P * H^T + R        (Innovation covariance)
       K = P * H^T * inv(S)       (Kalman Gain: how much to trust measurement vs prediction)
       x = x_hat + K * y          (Corrected state estimate)
       P = (I - K * H) * P        (Corrected uncertainty covariance)
"""

import numpy as np


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Converts a bounding box [x1, y1, x2, y2] into Kalman measurement format [u, v, s, r].

    Where:
        u: Center X coordinate
        v: Center Y coordinate
        s: Scale (bounding box area = width * height)
        r: Aspect ratio (width / height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Converts state vector [u, v, s, r, ...] back into bounding box [x1, y1, x2, y2].
    """
    u = x[0, 0]
    v = x[1, 0]
    s = x[2, 0]
    r = x[3, 0]

    # Handle edge case where scale or aspect ratio is non-positive
    if s <= 0:
        s = 1.0
    if r <= 0:
        r = 1.0

    w = np.sqrt(s * r)
    h = s / w

    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class KalmanBoxTrackerFilter:
    """
    7-dimensional state Kalman filter modeling linear bounding box dynamics:
        State vector x = [u, v, s, r, u_dot, v_dot, s_dot]^T
        Measurement z  = [u, v, s, r]^T
    """

    def __init__(self, initial_bbox: np.ndarray):
        """
        Initialize the Kalman filter with an initial detected bounding box.

        Args:
            initial_bbox: [x1, y1, x2, y2]
        """
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix F (constant velocity model)
        self.F = np.eye(self.dim_x, dtype=np.float32)
        # Position + Velocity * dt (dt = 1 frame)
        self.F[0, 4] = 1.0  # u = u + u_dot
        self.F[1, 5] = 1.0  # v = v + v_dot
        self.F[2, 6] = 1.0  # s = s + s_dot
        # Note: aspect ratio r is assumed constant, so r_dot = 0

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0  # Measure u
        self.H[1, 1] = 1.0  # Measure v
        self.H[2, 2] = 1.0  # Measure s
        self.H[3, 3] = 1.0  # Measure r

        # Measurement noise covariance R (uncertainty in YOLO detections)
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] *= 1.0    # u noise
        self.R[1, 1] *= 1.0    # v noise
        self.R[2, 2] *= 10.0   # s (area) noise
        self.R[3, 3] *= 1.0    # r noise

        # Process noise covariance Q (physical acceleration / velocity noise)
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01  # Velocity changes are smooth
        self.Q[2, 2] *= 0.01
        self.Q[6, 6] *= 0.0001

        # Initial state covariance P (high uncertainty in initial velocities)
        self.P = np.eye(self.dim_x, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0  # Velocities are unobserved initially
        self.P *= 10.0

        # Initialize state with the first observation
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.x[:4] = bbox_to_z(initial_bbox)

    def predict(self) -> np.ndarray:
        """
        Advance state estimation by 1 time step using the motion model.

        Returns:
            np.ndarray: Predicted bounding box [x1, y1, x2, y2]
        """
        # If scale + scale_velocity <= 0, clamp velocity to prevent negative area
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0

        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return x_to_bbox(self.x)

    def update(self, bbox: np.ndarray):
        """
        Incorporate new YOLO detection measurement to correct the state.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2]
        """
        z = bbox_to_z(bbox)

        # Innovation: y = z - H * x
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Updated covariance: P = (I - K * H) * P
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self) -> np.ndarray:
        """
        Returns current bounding box estimate [x1, y1, x2, y2].
        """
        return x_to_bbox(self.x)


if __name__ == "__main__":
    print("--- Testing Phase 5: Kalman Filter Bounding Box Motion & Occlusion ---")

    # Simulate a person walking horizontally across the frame at 10 pixels per frame
    # Frame 0: [100, 100, 150, 250] -> Center X = 125, Width = 50, Height = 150
    initial_box = np.array([100, 100, 150, 250], dtype=np.float32)
    kf = KalmanBoxTrackerFilter(initial_box)

    print("Initial box:", initial_box)

    # Simulate 5 frames with measurements (observed walking)
    for frame_idx in range(1, 6):
        # Motion: Person moves +10 pixels right each frame
        simulated_det = initial_box + np.array([frame_idx * 10, 0, frame_idx * 10, 0])
        pred_box = kf.predict()
        kf.update(simulated_det)
        est_box = kf.get_state()
        print(f"Frame {frame_idx:2d}: Measured Center-X={simulated_det[0]+25:5.1f} | Filter Estimated Center-X={est_box[0]+25:5.1f}")

    # Now simulate OCCLUSION for 3 frames (Frames 6, 7, 8: NO detections arrive!)
    print("\n--- Simulating 3 Frames of Occlusion (No detections available) ---")
    for frame_idx in range(6, 9):
        # We only call PREDICT, NO UPDATE!
        pred_box = kf.predict()
        expected_x = 125.0 + frame_idx * 10.0
        print(f"Frame {frame_idx:2d} (Occluded): Predicted Center-X={pred_box[0]+25:5.1f} | Expected Physical X={expected_x:5.1f}")
        # The filter prediction should closely track the true physical motion!
        assert np.isclose(pred_box[0] + 25, expected_x, atol=3.0), f"Prediction deviated too much during occlusion at frame {frame_idx}!"

    # Frame 9: Person reappears from behind pillar!
    reappear_det = initial_box + np.array([9 * 10, 0, 9 * 10, 0])
    pred_box = kf.predict()
    kf.update(reappear_det)
    est_box = kf.get_state()
    print(f"\nFrame  9 (Reappearance): Measured Center-X={reappear_det[0]+25:5.1f} | Updated Center-X={est_box[0]+25:5.1f}")
    assert np.isclose(est_box[0] + 25, 215.0, atol=2.0), "Failed to match after reappearance!"

    print("\nKalman Filter motion estimation and occlusion tests passed successfully!")
