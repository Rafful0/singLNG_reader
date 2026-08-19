"""Turns a noisy per-frame prediction into letters worth typing out.

The classifier judges every single frame on its own, so while your hand travels
from one letter to the next it happily reports whatever it sees on the way. The
stabilizer only accepts a letter once the last N frames agree on it, and only
emits it once: to repeat a letter you lower your hand (or move to another sign)
and sign it again, the same way a keyboard needs the key released first.
"""
from collections import deque
from typing import Deque, Optional


class LetterStabilizer:
    def __init__(self, stable_frames: int = 8, min_confidence: float = 0.7):
        if stable_frames < 1:
            raise ValueError('stable_frames must be at least 1')

        self.min_confidence = min_confidence
        # The last N frames, where None means "no hand, or not confident enough"
        self._recent: Deque[Optional[str]] = deque(maxlen=stable_frames)
        # The letter currently being held, already emitted. Cleared when the
        # hand is released, which is what makes a repeated letter possible.
        self._current: Optional[str] = None

    @property
    def window_size(self) -> int:
        return self._recent.maxlen or 0

    @property
    def current(self) -> Optional[str]:
        """The letter being held right now, or None while unsure."""
        return self._current

    def hold_progress(self) -> float:
        """0..1 - how much of the window agrees with the newest prediction.

        Only used to draw a progress bar, so you can watch a letter being
        "charged" instead of guessing why nothing happened yet.
        """
        if not self._recent or self._recent[-1] is None:
            return 0.0
        agreeing = sum(1 for prediction in self._recent if prediction == self._recent[-1])
        return agreeing / self.window_size

    def update(self, letter: Optional[str], confidence: float = 1.0) -> Optional[str]:
        """Feeds one frame in. Returns a letter only when it should be typed."""
        # Anything the model is not sure about counts as "no letter"
        self._recent.append(letter if letter is not None and confidence >= self.min_confidence else None)

        # Nothing is decided until the whole window agrees
        if len(self._recent) < self.window_size or any(item != self._recent[-1] for item in self._recent):
            return None

        stable = self._recent[-1]

        # A steady run of "no letter" means the hand was released, so the next
        # sign is allowed to be the same letter again.
        if stable is None:
            self._current = None
            return None

        # Same letter still being held: it was already typed, stay quiet
        if stable == self._current:
            return None

        self._current = stable
        return stable

    def reset(self) -> None:
        """Forgets the recent frames, e.g. after the sentence is cleared."""
        self._recent.clear()
        self._current = None
