"""Reads your sign alphabet from the camera and spells it out.

Every frame the trained classifier guesses a letter, but a guess is only added
to the sentence once several frames in a row agree on it (see
sign_language/smoothing.py). Hold a sign until the bar fills up, then move to
the next one; to sign the same letter twice, lower your hand in between.

    python interpret_signs.py

Keys: SPACE adds a space, BACKSPACE deletes, C clears, ESC quits.
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np

from sign_language import config, drawing
from sign_language.features import FEATURE_VERSION, landmarks_to_features
from sign_language.hand_stream import HandStream
from sign_language.smoothing import LetterStabilizer

KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = (8, 127)
KEY_CLEAR = ord('c')
KEY_NONE = 255


def load_model(path: Path):
    """Loads the trained bundle and refuses models built from older features."""
    if not path.exists():
        raise FileNotFoundError(
            f'No trained model at {path}. Record samples with collect_signs.py '
            'and run train_signs.py first.'
        )

    # joblib unpickles, which would run code hidden in a crafted file. This is
    # the standard way to persist a scikit-learn pipeline and the file is one
    # train_signs.py produced on this machine, so only load models you trained.
    bundle = joblib.load(path)
    if bundle.get('feature_version') != FEATURE_VERSION:
        raise ValueError(
            f'{path} was trained with feature version {bundle.get("feature_version")}, '
            f'but this code produces version {FEATURE_VERSION}. Run train_signs.py again.'
        )
    return bundle


def predict_letter(pipeline, features: np.ndarray) -> Tuple[str, float]:
    """Returns the most likely letter for one hand and how sure the model is."""
    sample = features.reshape(1, -1)

    # Models without probabilities still work, they just always report full
    # confidence and therefore rely only on the "N frames agree" rule.
    if hasattr(pipeline, 'predict_proba'):
        probabilities = pipeline.predict_proba(sample)[0]
        best = int(np.argmax(probabilities))
        return str(pipeline.classes_[best]), float(probabilities[best])

    return str(pipeline.predict(sample)[0]), 1.0


def draw_progress_bar(frame: np.ndarray, progress: float, origin: Tuple[int, int] = (10, 80),
                      size: Tuple[int, int] = (220, 16)) -> None:
    """Shows how close the current letter is to being accepted."""
    x, y = origin
    width, height = size
    cv2.rectangle(frame, (x, y), (x + width, y + height), drawing.BLACK, -1)
    cv2.rectangle(frame, (x, y), (x + int(width * min(1.0, progress)), y + height), drawing.GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), drawing.WHITE, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=config.CLASSIFIER_PATH,
                        help='Trained classifier (default: %(default)s)')
    parser.add_argument('--confidence', type=float, default=config.DEFAULT_MIN_CONFIDENCE,
                        help='Minimum probability to trust a frame (default: %(default)s)')
    parser.add_argument('--stable-frames', type=int, default=config.DEFAULT_STABLE_FRAMES,
                        help='Frames that must agree before a letter is typed (default: %(default)s)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: %(default)s)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        bundle = load_model(args.model)
    except (FileNotFoundError, ValueError) as error:
        print(f'Error: {error}')
        return 1

    pipeline = bundle['pipeline']
    print(f'Loaded {bundle.get("model_type", "?")} model for: {" ".join(bundle.get("labels", []))}')

    stabilizer = LetterStabilizer(stable_frames=args.stable_frames, min_confidence=args.confidence)
    sentence: List[str] = []

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

                # Mirror the frame, exactly like the recording script did: the
                # model only recognizes hands presented the same way.
                frame = cv2.flip(frame, 1)
                stream.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                hands = stream.hands()
                letter: Optional[str] = None
                confidence = 0.0

                if hands:
                    hand = hands[0]
                    letter, confidence = predict_letter(
                        pipeline, landmarks_to_features(hand.landmarks, hand.handedness)
                    )

                accepted = stabilizer.update(letter, confidence)
                if accepted:
                    sentence.append(accepted)
                    print(f'Letter: {accepted} ({confidence:.2f})')

                drawing.draw_hands(frame, [seen.points_2d for seen in hands])

                if letter is None:
                    status = 'Show one hand to the camera'
                else:
                    trusted = '' if confidence >= args.confidence else '  (unsure)'
                    status = f'{letter}  {confidence:.0%}{trusted}'
                drawing.draw_text_lines(frame, [status], (10, 40), scale=1.0)
                draw_progress_bar(frame, stabilizer.hold_progress())

                drawing.draw_bottom_banner(frame, ''.join(sentence) or '(SPACE, BACKSPACE, C to clear, ESC to quit)')
                cv2.imshow('Sign language interpreter', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == KEY_ESC:
                    break
                if key == KEY_SPACE:
                    sentence.append(' ')
                elif key in KEY_BACKSPACE:
                    if sentence:
                        sentence.pop()
                elif key == KEY_CLEAR:
                    sentence.clear()
                    # Forget the recent frames too, otherwise the letter still
                    # being held would not be typed again.
                    stabilizer.reset()
        finally:
            capture.release()
            cv2.destroyAllWindows()

    print(f'\nFinal text: {"".join(sentence)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
