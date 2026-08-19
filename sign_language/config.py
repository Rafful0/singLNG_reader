"""Shared configuration for the sign language interpreter.

Everything that is a "knob" (paths, thresholds, default sample counts) lives
here so the scripts stay focused on their own job.
"""
import string
from pathlib import Path

# Project root, i.e. the folder that holds main.py and the .task model
ROOT = Path(__file__).resolve().parent.parent

# The MediaPipe model. The same file used by the gesture demo is reused here:
# we only need the hand landmarks it returns, not its built-in gesture labels.
MODEL_PATH = ROOT / 'gesture_recognizer.task'

# Recorded samples (one row per captured hand) and the trained classifier
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
SAMPLES_PATH = DATA_DIR / 'sign_samples.csv'
CLASSIFIER_PATH = MODELS_DIR / 'sign_classifier.joblib'

# Labels the collector accepts by default: one key press per letter.
# Replace/extend this if your alphabet has extra signs (Ç, numbers, ...).
DEFAULT_ALPHABET = string.ascii_uppercase

# Recording defaults: how many frames make up one "take" of a letter, and how
# long you get to position your hand before the take starts.
DEFAULT_SAMPLES_PER_TAKE = 40
DEFAULT_COUNTDOWN_SECONDS = 2.0

# Live interpretation defaults.
# A prediction is only trusted above this probability...
DEFAULT_MIN_CONFIDENCE = 0.70
# ...and only after the same letter wins this many frames in a row, which
# filters out the noisy frames produced while the hand travels between signs.
DEFAULT_STABLE_FRAMES = 8
