"""Turns raw hand landmarks into a fixed-size numeric vector for scikit-learn.

MediaPipe gives 21 landmarks per hand in *normalized image* coordinates, so the
same handshape produces very different numbers depending on where the hand is,
how far it is from the camera, how tilted it is and which hand you used. A
classifier trained on those raw numbers would mostly learn "where you stood
while recording", so every sample is first put into a canonical pose:

  1. left/right hands are mirrored onto the same chirality
  2. the wrist is moved to the origin      -> position no longer matters
  3. the hand is rotated to point "up"     -> tilt no longer matters
  4. the hand is scaled to a unit span     -> distance to camera no longer matters

What is left describes the *shape* of the hand, which is exactly what tells one
letter apart from another.
"""
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Indices of the landmarks MediaPipe returns (see the hand landmark map in the
# MediaPipe docs). Only the ones this module needs are named.
WRIST = 0
MIDDLE_MCP = 9  # knuckle of the middle finger: a stable "up" direction
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
FINGERTIP_NAMES = ('thumb', 'index', 'middle', 'ring', 'pinky')

NUM_LANDMARKS = 21

# Bumped whenever the maths below changes, so a model trained with an older
# feature layout is rejected instead of silently producing nonsense.
FEATURE_VERSION = 1

Landmark = Sequence[float]  # (x, y, z), or any object indexable like one


def _to_array(landmarks: Iterable[Landmark]) -> np.ndarray:
    """Copies the landmarks into a plain (21, 3) float array."""
    points = np.array([[float(lm[0]), float(lm[1]), float(lm[2])] for lm in landmarks], dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f'Expected {NUM_LANDMARKS} landmarks with 3 coordinates, got {points.shape}')
    return points


def normalize_landmarks(landmarks: Iterable[Landmark], handedness: str = 'Right') -> np.ndarray:
    """Returns the landmarks in the canonical pose described at the top.

    `handedness` is the label MediaPipe reports for the hand ('Left'/'Right').
    Left hands are mirrored so that a letter signed with either hand lands on
    the same point of the feature space, which roughly halves how much you have
    to record.
    """
    points = _to_array(landmarks)

    # 1. Mirror left hands. Note the demo shows a flipped (mirror-like) frame,
    # so this label is about the image, not about your actual hand - all that
    # matters is that collecting and predicting use the same convention.
    if handedness.lower().startswith('l'):
        points[:, 0] = -points[:, 0]

    # 2. Move the wrist to the origin: the hand's position in frame is now gone.
    points -= points[WRIST]

    # 3. Rotate around the image plane so the wrist -> middle knuckle vector
    # points straight up. Image coordinates grow downwards, so "up" is -y.
    dx, dy = points[MIDDLE_MCP, 0], points[MIDDLE_MCP, 1]
    current_angle = math.atan2(dy, dx)
    rotation = -math.pi / 2 - current_angle
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    rotated_x = points[:, 0] * cos_r - points[:, 1] * sin_r
    rotated_y = points[:, 0] * sin_r + points[:, 1] * cos_r
    points[:, 0], points[:, 1] = rotated_x, rotated_y

    # 4. Scale so the farthest landmark from the wrist sits at distance 1. Only
    # x and y are used because the depth MediaPipe estimates from a single
    # camera is a rough guess and would make the scale jitter.
    span = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if span > 1e-6:  # a degenerate hand would otherwise divide by zero
        points /= span

    return points


def landmarks_to_features(landmarks: Iterable[Landmark], handedness: str = 'Right') -> np.ndarray:
    """Builds the vector that is stored in the dataset and fed to the model.

    It is the canonical pose (every landmark except the wrist, which is always
    at the origin) plus the distances between fingertips, which make "how open
    is the hand" explicit instead of something the model has to infer.
    """
    points = normalize_landmarks(landmarks, handedness)

    # Landmarks 1..20 as x, y, z triplets
    coordinates = points[1:].reshape(-1)

    # Pairwise fingertip distances, measured in the image plane for the same
    # reason the scaling above ignores z.
    distances: List[float] = []
    for i, first in enumerate(FINGERTIPS):
        for second in FINGERTIPS[i + 1:]:
            distances.append(float(np.linalg.norm(points[first, :2] - points[second, :2])))

    return np.concatenate([coordinates, np.array(distances, dtype=np.float64)])


def feature_names() -> Tuple[str, ...]:
    """Column names for the vector above, used as the CSV header."""
    names: List[str] = []
    for index in range(1, NUM_LANDMARKS):
        names.extend([f'x{index}', f'y{index}', f'z{index}'])
    for i, first in enumerate(FINGERTIP_NAMES):
        for second in FINGERTIP_NAMES[i + 1:]:
            names.append(f'dist_{first}_{second}')
    return tuple(names)


# Handy constant: how wide one sample is (60 coordinates + 10 distances = 70)
FEATURE_SIZE = len(feature_names())
