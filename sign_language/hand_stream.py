"""A small wrapper around MediaPipe's recognizer in LIVE_STREAM mode.

Live-stream mode is asynchronous: you push frames in and results come back
later, on a MediaPipe worker thread. That means two things every camera loop
has to deal with, and that are handled here once instead of in each script:

  * the results arrive on another thread, so they need a lock
  * pushing frames faster than they can be processed builds up lag, so a new
    frame is only sent once the previous result came back (back-pressure)

The gesture recognizer model is reused simply because it is the model this
repository already ships; the sign language feature only reads the hand
landmarks that come with every result.
"""
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

from . import config

# Shortcuts to the tasks API, same as in the gesture demo
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandObservation:
    """One hand seen in one frame."""

    # 21 landmarks as (x, y, z) in normalized image coordinates
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    # 'Left' or 'Right' as reported by MediaPipe for the frame it was given
    handedness: str = 'Right'
    # The label from the built-in gesture model, kept for the original demo
    gesture: Optional[str] = None
    gesture_score: float = 0.0

    @property
    def points_2d(self) -> List[Tuple[float, float]]:
        """The landmarks without depth, which is all the drawing code needs."""
        return [(x, y) for x, y, _ in self.landmarks]


class HandStream:
    """Feeds camera frames to MediaPipe and hands back the latest hands seen."""

    def __init__(self, num_hands: int = 1, model_path: Path = config.MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f'Model not found at {model_path}. Download gesture_recognizer.task '
                'and place it in the project root (see the README).'
            )

        self._options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=num_hands,
        )
        self._recognizer: Optional[GestureRecognizer] = None

        # Guards everything below it: the callback runs on another thread
        self._lock = threading.Lock()
        self._hands: List[HandObservation] = []
        self._awaiting_result = False
        # MediaPipe rejects timestamps that do not strictly increase
        self._last_timestamp_ms = -1

    def __enter__(self) -> 'HandStream':
        self._recognizer = GestureRecognizer.create_from_options(self._options)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def submit(self, rgb_frame: np.ndarray) -> bool:
        """Sends a frame for recognition. Returns False if one is still in flight."""
        if self._recognizer is None:
            raise RuntimeError('Use HandStream as a context manager: `with HandStream() as stream:`')

        with self._lock:
            if self._awaiting_result:
                return False
            self._awaiting_result = True
            timestamp_ms = max(int(time.perf_counter() * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._recognizer.recognize_async(mp_image, timestamp_ms)
        return True

    def hands(self) -> List[HandObservation]:
        """The hands from the most recent result (empty list when none were seen)."""
        with self._lock:
            return list(self._hands)

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        """Called by MediaPipe on its own thread once a frame has been processed."""
        observations: List[HandObservation] = []

        hand_landmarks = getattr(result, 'hand_landmarks', None) or []
        handedness = getattr(result, 'handedness', None) or []
        gestures = getattr(result, 'gestures', None) or []

        for index, landmarks in enumerate(hand_landmarks):
            # The parallel lists are not guaranteed to be filled, so read them
            # defensively and fall back to sane defaults.
            hand_label = handedness[index][0].category_name if index < len(handedness) and handedness[index] else 'Right'
            gesture_name, gesture_score = None, 0.0
            if index < len(gestures) and gestures[index]:
                gesture_name = gestures[index][0].category_name
                gesture_score = float(gestures[index][0].score)

            observations.append(
                HandObservation(
                    landmarks=[(lm.x, lm.y, lm.z) for lm in landmarks],
                    handedness=hand_label,
                    gesture=gesture_name,
                    gesture_score=gesture_score,
                )
            )

        with self._lock:
            self._hands = observations
            self._awaiting_result = False
