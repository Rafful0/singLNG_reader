"""Overlay helpers shared by the demo, the collector and the interpreter.

Nothing here is required to make the recognition work - it just makes the
camera window readable while you use it.
"""
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                 # Palm base connection
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def draw_hands(frame: np.ndarray, hands: Iterable[Sequence[Sequence[float]]],
               line_color: Tuple[int, int, int] = GREEN,
               point_color: Tuple[int, int, int] = RED) -> None:
    """Draws the skeleton of every hand on top of the frame.

    `hands` holds one sequence of normalized (x, y) points per hand - exactly
    what MediaPipe returns, so it can be drawn without converting anything but
    the scale.
    """
    height, width = frame.shape[:2]

    for hand in hands:
        # Normalized coordinates (0..1) into pixels
        points = [(int(point[0] * width), int(point[1] * height)) for point in hand]

        # Each line is drawn twice: a thick dark outline first so the colored
        # line stays visible over bright backgrounds.
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], BLACK, 10, cv2.LINE_AA)
                cv2.line(frame, points[start], points[end], line_color, 6, cv2.LINE_AA)

        for x_px, y_px in points:
            cv2.circle(frame, (x_px, y_px), 10, BLACK, -1)
            cv2.circle(frame, (x_px, y_px), 6, point_color, -1)


def draw_text_lines(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (10, 30),
                    scale: float = 0.8, color: Tuple[int, int, int] = WHITE, line_height: int = 30) -> None:
    """Writes one line of text per entry, with an outline for readability."""
    x, y = origin
    for index, line in enumerate(lines):
        position = (x, y + index * line_height)
        cv2.putText(frame, line, position, FONT, scale, BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, FONT, scale, color, 2, cv2.LINE_AA)


def draw_bottom_banner(frame: np.ndarray, text: str, height: int = 70, scale: float = 1.1,
                       color: Tuple[int, int, int] = WHITE) -> None:
    """Draws a dark strip at the bottom of the frame holding a single line.

    Used for the sentence being spelled out; the text is trimmed from the left
    so the most recent letters always stay on screen.
    """
    frame_height, frame_width = frame.shape[:2]
    top = frame_height - height

    # Semi-transparent strip, so the camera image still shows through
    strip = frame[top:frame_height].copy()
    cv2.rectangle(strip, (0, 0), (frame_width, height), BLACK, -1)
    cv2.addWeighted(strip, 0.6, frame[top:frame_height], 0.4, 0, frame[top:frame_height])

    visible = _fit_text(text, frame_width - 20, scale)
    cv2.putText(frame, visible, (10, top + int(height * 0.65)), FONT, scale, color, 2, cv2.LINE_AA)


def _fit_text(text: str, max_width_px: int, scale: float) -> str:
    """Drops characters from the left until the text fits the given width."""
    visible = text
    while visible:
        (text_width, _), _ = cv2.getTextSize(visible, FONT, scale, 2)
        if text_width <= max_width_px:
            break
        visible = visible[1:]
    return visible


def format_counts(counts, alphabet: str, per_line: int = 13) -> List[str]:
    """Turns per-letter sample counts into short lines like `A:40 B:12 C:0`."""
    entries = [f'{letter}:{counts.get(letter, 0)}' for letter in alphabet]
    return [' '.join(entries[index:index + per_line]) for index in range(0, len(entries), per_line)]
