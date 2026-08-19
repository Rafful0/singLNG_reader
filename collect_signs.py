"""Records what each letter of your sign alphabet looks like.

Point the camera at yourself, hold a letter and press that letter's key: the
script waits a moment (so you can settle into the sign), then stores a burst of
samples in data/sign_samples.csv. Repeat for every letter, ideally recording
each one a few times with slightly different angles and distances - the model
can only be as varied as what you show it.

    python collect_signs.py

Keys: A-Z record that letter, BACKSPACE undoes the last take, ESC quits.
"""
import argparse
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from sign_language import config, dataset, drawing
from sign_language.features import landmarks_to_features
from sign_language.hand_stream import HandStream

# Key codes OpenCV reports for the control keys
KEY_ESC = 27
KEY_BACKSPACE = (8, 127)  # differs between platforms
KEY_NONE = 255  # returned by waitKey when nothing was pressed


class TakeRecorder:
    """The little state machine behind one take: countdown, then capture."""

    def __init__(self, samples_per_take: int, countdown_seconds: float):
        self.samples_per_take = samples_per_take
        self.countdown_seconds = countdown_seconds

        self.label: Optional[str] = None
        self.vectors: List[np.ndarray] = []
        self._starts_at = 0.0
        # Landmarks of the last stored sample. Results only change when
        # MediaPipe finishes a new frame, so this avoids storing the very same
        # hand several times, which would look like data but add nothing.
        self._last_landmarks: Optional[list] = None

    @property
    def is_active(self) -> bool:
        return self.label is not None

    @property
    def is_counting_down(self) -> bool:
        return self.is_active and time.monotonic() < self._starts_at

    @property
    def is_complete(self) -> bool:
        return len(self.vectors) >= self.samples_per_take

    def start(self, label: str) -> None:
        self.label = label
        self.vectors = []
        self._starts_at = time.monotonic() + self.countdown_seconds
        self._last_landmarks = None

    def cancel(self) -> None:
        self.label = None
        self.vectors = []
        self._last_landmarks = None

    def seconds_left(self) -> float:
        return max(0.0, self._starts_at - time.monotonic())

    def capture(self, hand) -> None:
        """Stores one sample if this hand was not stored already."""
        if not self.is_active or self.is_counting_down or self.is_complete:
            return
        if hand.landmarks == self._last_landmarks:
            return

        self.vectors.append(landmarks_to_features(hand.landmarks, hand.handedness))
        self._last_landmarks = hand.landmarks

    def status_lines(self, hand_visible: bool) -> List[str]:
        """One line describing what the recorder is waiting for right now."""
        if not self.is_active:
            return ['Press a letter key to record it']
        if self.is_counting_down:
            return [f'Get ready for "{self.label}"... {self.seconds_left():.1f}s']
        if not hand_visible:
            return [f'Recording "{self.label}" - show your hand to the camera']
        return [f'Recording "{self.label}": {len(self.vectors)}/{self.samples_per_take}']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', type=int, default=config.DEFAULT_SAMPLES_PER_TAKE,
                        help='How many samples one take records (default: %(default)s)')
    parser.add_argument('--delay', type=float, default=config.DEFAULT_COUNTDOWN_SECONDS,
                        help='Seconds between pressing the key and recording (default: %(default)s)')
    parser.add_argument('--alphabet', default=config.DEFAULT_ALPHABET,
                        help='Characters that can be recorded (default: A-Z)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: %(default)s)')
    parser.add_argument('--data', type=Path, default=config.SAMPLES_PATH,
                        help='CSV the samples are appended to (default: %(default)s)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alphabet = args.alphabet.upper()
    recorder = TakeRecorder(args.samples, args.delay)

    # Kept in memory so the on-screen counters do not re-read the CSV every frame
    counts = dataset.label_counts(args.data)
    last_take_size = 0
    message = ''

    with HandStream(num_hands=1) as stream:
        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            print(f'Error: could not open camera {args.camera}')
            return 1

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # Mirror the frame so moving right moves right on screen
                frame = cv2.flip(frame, 1)
                stream.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                hands = stream.hands()
                hand = hands[0] if hands else None

                if hand is not None:
                    recorder.capture(hand)

                # A finished take is written in one go, so BACKSPACE can undo
                # exactly the take you just recorded.
                if recorder.is_complete:
                    dataset.append_samples([recorder.label] * len(recorder.vectors), recorder.vectors, args.data)
                    counts[recorder.label] += len(recorder.vectors)
                    last_take_size = len(recorder.vectors)
                    message = f'Saved {last_take_size} samples for "{recorder.label}"'
                    print(message)
                    recorder.cancel()

                drawing.draw_hands(frame, [seen.points_2d for seen in hands])
                drawing.draw_text_lines(frame, recorder.status_lines(hand is not None), (10, 30), scale=0.8)
                drawing.draw_text_lines(frame, drawing.format_counts(counts, alphabet), (10, 70),
                                        scale=0.6, line_height=24)
                drawing.draw_bottom_banner(frame, message or 'Letter: record | BACKSPACE: undo | ESC: quit',
                                           scale=0.7)

                cv2.imshow('Record signs', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == KEY_ESC:
                    break
                if key in KEY_BACKSPACE:
                    # While a take is running BACKSPACE aborts it, otherwise it
                    # deletes the samples written by the previous take.
                    if recorder.is_active:
                        recorder.cancel()
                        message = 'Take cancelled'
                    elif last_take_size:
                        removed = dataset.remove_last_samples(last_take_size, args.data)
                        counts = dataset.label_counts(args.data)
                        last_take_size = 0
                        message = f'Removed the last {removed} samples'
                    else:
                        message = 'Nothing left to undo'
                    print(message)
                elif key != KEY_NONE:
                    pressed = chr(key).upper()
                    if pressed in alphabet and not recorder.is_active:
                        recorder.start(pressed)
                        message = ''
        finally:
            capture.release()
            cv2.destroyAllWindows()

    total = sum(counts.values())
    print(f'\n{total} samples across {len(counts)} labels in {args.data}')
    print('Next step: python train_signs.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
