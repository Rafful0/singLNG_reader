# Gesture Recognizer Demo
I recently got into AI and ML and had trouble finding good recent examples of using MediaPipe to recognize
Hand gestures. Even [Googles official docs](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#live-stream) wasn't very clear in my opinion. So alas, after a bit of research I have a straightforward example of using Google's MediaPipe that *hopefully* anyone can follow.


This repository contains a minimal example that demonstrates using MediaPipe's
GestureRecognizer in live-stream mode (camera input) and it ships a **sign language interpreter** you teach yourself: you record what each letter looks like *with your own hands*, train a scikit-learn
classifier on those recordings, and then spell words at the camera. Nothing is
hard-coded to a specific alphabet — record ASL, Libras or a set of signs you
made up, the pipeline does not care.

<img width="962" height="568" alt="Screenshot 2025-09-27 at 9 24 59 AM" src="https://github.com/user-attachments/assets/b00099a1-a8a4-42b4-94d3-3934a424160b" />

## Quickstart

1. Install Python 3.10 and create a virtual environment (recommended):

```bash
python3.10 -m venv my_venv
source my_venv/bin/activate
```

On Windows, activate it with `my_venv\Scripts\activate` instead.

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download the MediaPipe gesture recognizer [.task file model here](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models) and place it into the root of the repository. This default model can recognize seven classes (i.e. 👍, 👎, ✌️, ☝️, ✊, 👋, 🤟) in one or two hands

## Sign language interpreter

The same `.task` model is reused here — the interpreter only needs the 21 hand
landmarks that come with every result, and a scikit-learn classifier learns the
letters on top of them. Three steps, one script each.

### 1. Record what each letter means

```bash
python collect_signs.py
```

Hold a sign and press that letter's key. After a short countdown the script
records a burst of samples into `data/sign_samples.csv` and shows a counter per
letter, so you always know what is still missing.

| Key | Action |
| --- | --- |
| `A`–`Z` | Record a take of that letter |
| `BACKSPACE` | Cancel the running take, or undo the last saved one |
| `ESC` | Quit |

Aim for **at least 40 samples per letter**, recorded in a few separate takes:
move a bit closer and further away, rotate your hand slightly, use both hands,
change the background. The classifier can only be as varied as what you show
it, and one long take of a perfectly still hand teaches it very little.

Useful flags: `--samples 60` (samples per take), `--delay 3` (countdown),
`--alphabet ABC` (restrict or extend the recordable characters), `--camera 1`.

### 2. Train the model

```bash
python train_signs.py
```

### 3. Interpret

```bash
python interpret_signs.py
```

Hold a sign until the green bar fills up and the letter is appended to the
sentence at the bottom of the window. To sign the same letter twice in a row,
lower your hand in between — like releasing a key before pressing it again.

| Key | Action |
| --- | --- |
| `SPACE` | Add a space |
| `BACKSPACE` | Delete the last character |
| `C` | Clear the sentence |
| `ESC` | Quit |

Getting letters you did not sign? Raise `--confidence 0.85` or `--stable-frames 12`.
Getting nothing at all? Lower them.

### How it works

The same handshape produces wildly different landmark coordinates depending on
where you stand, how far you are from the camera and how you tilt your wrist, so
every sample is put into a canonical pose before it reaches the model
(`sign_language/features.py`): left hands are mirrored, the wrist is moved to
the origin, the hand is rotated to point up and scaled to a unit span. What
remains describes the *shape* of the hand — 60 coordinates plus 10 fingertip
distances, 70 features per sample.

Live predictions are then smoothed (`sign_language/smoothing.py`): a letter is
only typed once several consecutive frames agree on it and the model is
confident enough, which is what stops the sentence from filling with garbage
while your hand travels between signs.

### Limitations

Only **static** handshapes are recognized: each prediction looks at one frame, so
letters defined by movement (J and Z in ASL, H, K, X, Y and Z in Libras) cannot
be told apart from their starting pose. They would need a model that reads a
sequence of frames instead of a single one.

## Files
- `main.py` — Live camera demo using MediaPipe GestureRecognizer in
  LIVE_STREAM mode.
- `collect_signs.py` — Records samples for each letter of your alphabet.
- `train_signs.py` — Trains and evaluates the scikit-learn classifier.
- `interpret_signs.py` — Live interpretation: spells out the signs you make.
- `sign_language/` — The shared pieces: camera stream wrapper, feature
  extraction, dataset storage, prediction smoothing and drawing helpers.
- `data/sign_samples.csv` — (created by you) Your recordings, one row per sample.
- `models/sign_classifier.joblib` — (created by you) The trained classifier.
- `gesture_recognizer.task` — (not included) The model file expected by the
  demo. Obtain a model from [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models) or export one compatible with MediaPipe
  Tasks.
- `requirements.txt` — Python dependencies used by this project.


## License
mediapipe_gesture_recognition is under [Apache v2 license](LICENSE).
