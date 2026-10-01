# Dataset Directory & Setup Instructions

This directory houses test video sequences and standard benchmark datasets for Multi-Object Tracking.

## MOT17 / MOT20 Benchmark Setup

The MOT (Multiple Object Tracking) challenge datasets provide standardized pedestrian tracking sequences with ground truth annotations.

### How to Download MOT17 / MOT20:
1. Visit the official MOT challenge website: https://motchallenge.net/
2. Download the MOT17 or MOT20 dataset (`MOT17.zip` or `MOT20.zip`).
3. Extract the contents so that the directory structure is organized as:

```text
data/
└── MOT17/
    ├── train/
    │   ├── MOT17-02-FRCNN/
    │   │   ├── img1/         # Frame images (e.g. 000001.jpg, 000002.jpg)
    │   │   ├── gt/
    │   │   │   └── gt.txt    # Ground truth annotations
    │   │   └── seqinfo.ini   # Sequence metadata (fps, length, width, height)
    │   ├── MOT17-04-FRCNN/
    │   └── ...
    └── test/
```

### Ground Truth Format (`gt.txt`):
Each line represents an annotation in the following format:
`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>`

- `class`: 1 indicates pedestrian/person.
- `visibility`: float between 0 and 1 indicating how visible the person is (crucial for evaluating occlusion!).

## Sample Sequences for Testing Occlusion

For fast unit testing and verifying the pipeline without downloading hundreds of gigabytes, a synthetic or sample sequence is provided or can be generated in `data/sample/`.
