"""
Realistic Retail Occlusion Video & Ground Truth Generator
Project: AIML-03 - Multi-Object Tracking Under Occlusion

Generates a realistic test sequence where two shoppers move across a retail floor.
Shopper 1 passes behind a large architectural pillar, experiencing complete occlusion for 7 frames.
Simultaneously exports standard MOT benchmark ground-truth annotations (gt.txt)
for quantitative MOTA, False Positive, False Negative, and ID Switch evaluation.
"""

import os
import cv2
import numpy as np
from ultralytics.utils import ASSETS


def generate_retail_sequence(output_dir: str = "data/sample/retail_seq", num_frames: int = 100):
    """
    Generates video frames and gt.txt in MOT17 dataset format.
    """
    img_dir = os.path.join(output_dir, "img1")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    bus_path = os.path.join(ASSETS, "bus.jpg")
    bus = cv2.imread(bus_path)

    # Crop real person 1 (Person A) and person 2 (Person B) from bus image
    # Person 1: [48, 398, 245, 902] -> width ~ 197, height ~ 504
    # Person 2: [669, 392, 809, 877] -> width ~ 140, height ~ 485
    p1 = bus[398:880, 52:240]
    p2 = bus[395:870, 672:805]

    # Resize to realistic surveillance proportions
    target_h, target_w1 = 180, 70
    p1_resized = cv2.resize(p1, (target_w1, target_h))
    target_w2 = 60
    p2_resized = cv2.resize(p2, (target_w2, target_h))

    # Canvas dimensions
    width, height = 960, 540

    # Obstacle (Ceiling support pillar in middle of floor)
    pillar_x1, pillar_y1 = 440, 100
    pillar_x2, pillar_y2 = 530, 460

    gt_records = []
    # MOT gt.txt format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    print(f"[DataGen] Synthesizing {num_frames} frames with real human features and retail occlusion...")

    for f in range(1, num_frames + 1):
        # Create tiled retail floor
        frame = np.full((height, width, 3), (225, 228, 232), dtype=np.uint8)

        # Floor tile grid
        for gy in range(0, height, 40):
            cv2.line(frame, (0, gy), (width, gy), (205, 208, 212), 1)
        for gx in range(0, width, 40):
            cv2.line(frame, (gx, 0), (gx, height), (205, 208, 212), 1)

        # Ambient store walls
        cv2.rectangle(frame, (0, 0), (width, 70), (180, 185, 190), -1)
        cv2.line(frame, (0, 70), (width, 70), (140, 145, 150), 2)
        cv2.putText(frame, "RETAIL OUTLET - CEILING CAMERA 03", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        # Motion trajectories:
        # Person A (ID 1): walks from Left (X=80) to Right (X=800)
        # Person B (ID 2): walks from Right (X=840) to Left (X=100)
        t = (f - 1) / float(num_frames - 1)
        x1_pos = int(80 + t * 720)
        y1_pos = 260

        x2_pos = int(840 - t * 740)
        y2_pos = 220

        # Draw Person B on floor
        frame[y2_pos:y2_pos + target_h, x2_pos:x2_pos + target_w2] = p2_resized
        gt_records.append(f"{f},2,{x2_pos},{y2_pos},{target_w2},{target_h},1,1,1.0")

        # Determine occlusion for Person A
        # Person A is behind pillar when x1_pos is between pillar_x1 - 30 and pillar_x2
        is_occluded = (pillar_x1 - 35 <= x1_pos <= pillar_x2 - 15)
        visibility = 0.0 if is_occluded else 1.0

        if not is_occluded:
            frame[y1_pos:y1_pos + target_h, x1_pos:x1_pos + target_w1] = p1_resized
            gt_records.append(f"{f},1,{x1_pos},{y1_pos},{target_w1},{target_h},1,1,{visibility:.1f}")
        else:
            # When occluded, record in gt.txt with visibility 0.0
            gt_records.append(f"{f},1,{x1_pos},{y1_pos},{target_w1},{target_h},1,1,0.0")

        # Draw the Pillar in the foreground (occluding anything behind it)
        cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (85, 90, 100), -1)
        cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (40, 45, 55), 3)
        cv2.putText(frame, "STORE PILLAR", (pillar_x1 + 6, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Save frame image
        frame_name = f"{f:06d}.jpg"
        cv2.imwrite(os.path.join(img_dir, frame_name), frame)

    # Write ground truth gt.txt
    gt_file = os.path.join(gt_dir, "gt.txt")
    with open(gt_file, "w") as f_out:
        f_out.write("\n".join(gt_records) + "\n")

    # Write seqinfo.ini
    seqinfo_file = os.path.join(output_dir, "seqinfo.ini")
    with open(seqinfo_file, "w") as f_seq:
        f_seq.write(f"""[Sequence]
name=retail_occlusion_seq
imDir=img1
frameRate=25
seqLength={num_frames}
imWidth={width}
imHeight={height}
imExt=.jpg
""")

    print(f"[DataGen] Sequence generation complete!")
    print(f"  Images: {img_dir} ({num_frames} frames)")
    print(f"  Ground Truth: {gt_file}")
    return output_dir


if __name__ == "__main__":
    generate_retail_sequence()
