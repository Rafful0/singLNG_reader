"""Sign language interpreter built on top of the MediaPipe hand landmarks.

The pipeline has three steps, one script each at the repository root:

    collect_signs.py  -> record what each letter looks like  (data/sign_samples.csv)
    train_signs.py    -> fit a scikit-learn classifier       (models/sign_classifier.joblib)
    interpret_signs.py-> spell out letters from the camera

This package holds the pieces they share: reading the camera stream, turning
landmarks into features, storing samples, smoothing predictions and drawing.
"""
from . import config, dataset, drawing, features, hand_stream, smoothing

__all__ = ['config', 'dataset', 'drawing', 'features', 'hand_stream', 'smoothing']
