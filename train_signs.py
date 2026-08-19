"""Trains the scikit-learn classifier that recognizes your recorded letters.

It reads data/sign_samples.csv, holds part of it back to measure how well the
model does on hands it never saw, prints a report and finally saves a model
fitted on *all* samples to models/sign_classifier.joblib.

    python train_signs.py
    python train_signs.py --model rf    # if the default SVM struggles

The default is a support vector machine on standardized features: with a few
dozen samples per letter and only 70 features it trains in seconds, handles the
non-linear boundaries between similar handshapes well, and gives calibrated
probabilities, which the interpreter uses to stay quiet when unsure.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sign_language import config, dataset
from sign_language.features import FEATURE_VERSION, feature_names

# Fewer samples than this per letter and there is nothing left to evaluate on
# after splitting, so training stops instead of reporting a meaningless score.
MIN_SAMPLES_PER_LABEL = 5

# Below this you get a warning, not an error: it still trains, just poorly.
RECOMMENDED_SAMPLES_PER_LABEL = 20


def build_pipeline(model_type: str, seed: int, calibration_folds: int = 5) -> Pipeline:
    """Creates the scaler + classifier pipeline for the chosen model.

    The scaler is part of the pipeline on purpose: it is fitted on the training
    fold only, so cross validation cannot peek at the test data through it, and
    the saved model carries its own scaling at prediction time.
    """
    if model_type == 'svm':
        # `balanced` weights keep a letter you recorded less often from being
        # ignored. An SVM has no probabilities of its own, so it is wrapped in a
        # calibrator that turns its decision values into the confidence the
        # interpreter needs to stay quiet when unsure.
        classifier = CalibratedClassifierCV(
            SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', random_state=seed),
            method='sigmoid', ensemble=False, cv=calibration_folds,
        )
    elif model_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                            n_jobs=-1, random_state=seed)
    elif model_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    else:  # argparse already restricts this, kept for anyone calling directly
        raise ValueError(f'Unknown model type: {model_type}')

    return Pipeline([('scaler', StandardScaler()), ('classifier', classifier)])


def describe_dataset(labels: np.ndarray) -> Counter:
    """Prints how many samples each letter has and warns about thin ones."""
    counts = Counter(labels.tolist())
    print(f'Loaded {len(labels)} samples across {len(counts)} labels:')
    for label, count in sorted(counts.items()):
        warning = '  <- record more of this one' if count < RECOMMENDED_SAMPLES_PER_LABEL else ''
        print(f'  {label}: {count}{warning}')
    return counts


def print_confusion_matrix(true_labels: np.ndarray, predicted: np.ndarray, labels: List[str]) -> None:
    """Prints the confusion matrix, i.e. which letters get mixed up with which."""
    matrix = confusion_matrix(true_labels, predicted, labels=labels)
    width = max(3, max(len(label) for label in labels))

    print('\nConfusion matrix (rows = signed, columns = predicted):')
    print(' ' * (width + 1) + ' '.join(label.rjust(width) for label in labels))
    for label, row in zip(labels, matrix):
        print(label.rjust(width) + ' ' + ' '.join(str(value).rjust(width) for value in row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('svm', 'rf', 'knn'), default='svm',
                        help='Classifier to train (default: %(default)s)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction held back to evaluate the model (default: %(default)s)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Cross validation folds (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
    parser.add_argument('--data', type=Path, default=config.SAMPLES_PATH,
                        help='Recorded samples (default: %(default)s)')
    parser.add_argument('--out', type=Path, default=config.CLASSIFIER_PATH,
                        help='Where to save the trained model (default: %(default)s)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        features, labels = dataset.load_dataset(args.data)
    except (FileNotFoundError, ValueError) as error:
        print(f'Error: {error}')
        return 1

    counts = describe_dataset(labels)

    if len(counts) < 2:
        print('\nError: at least two different letters are needed to train a classifier.')
        return 1

    smallest_label, smallest_count = min(counts.items(), key=lambda item: item[1])
    if smallest_count < MIN_SAMPLES_PER_LABEL:
        print(f'\nError: "{smallest_label}" has only {smallest_count} sample(s), at least '
              f'{MIN_SAMPLES_PER_LABEL} are needed. Record another take of it.')
        return 1

    # Hold data back *before* fitting anything: the score below is only honest
    # if the model has never seen these hands.
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels,
    )

    # Every fold count below is capped by the thinnest letter *in the training
    # split*, because a fold cannot contain more samples of a letter than exist.
    train_smallest = min(Counter(train_labels.tolist()).values())
    folds = max(2, min(args.folds, train_smallest))
    # The SVM calibrator splits again inside each of those folds, so it gets
    # what is left after the outer split takes its share.
    calibration_folds = max(2, min(5, int(train_smallest * (folds - 1) / folds)))

    pipeline = build_pipeline(args.model, args.seed, calibration_folds)

    # Cross validation on the training part gives a stability estimate: a wide
    # spread means the result depends a lot on which samples were used.
    scores = cross_val_score(pipeline, train_features, train_labels,
                             cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed))
    print(f'\nCross validation ({folds} folds): {scores.mean():.3f} +/- {scores.std():.3f}')

    pipeline.fit(train_features, train_labels)
    predicted = pipeline.predict(test_features)
    print(f'Held-out accuracy: {accuracy_score(test_labels, predicted):.3f}\n')
    print(classification_report(test_labels, predicted, zero_division=0))

    print_confusion_matrix(test_labels, predicted, sorted(counts))

    # Now that the numbers are known, refit on everything: the model that gets
    # saved should use every sample you took the trouble to record.
    pipeline.fit(features, labels)

    bundle: Dict[str, object] = {
        'pipeline': pipeline,
        'labels': sorted(counts),
        'model_type': args.model,
        'feature_version': FEATURE_VERSION,
        'feature_names': feature_names(),
        'num_samples': int(len(labels)),
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f'\nSaved model to {args.out}')
    print('Next step: python interpret_signs.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
