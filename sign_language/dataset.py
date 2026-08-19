"""Reading and writing the recorded samples.

The dataset is a plain CSV: one row per captured hand, first column the letter,
remaining columns the feature vector. A CSV keeps the recordings inspectable
(and fixable) with any text editor or spreadsheet, which matters a lot when you
are the one producing the data.
"""
import csv
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from . import config
from .features import FEATURE_SIZE, feature_names

LABEL_COLUMN = 'label'


def _header() -> List[str]:
    return [LABEL_COLUMN, *feature_names()]


def append_samples(labels: Sequence[str], vectors: Sequence[np.ndarray], path: Path = config.SAMPLES_PATH) -> None:
    """Adds recorded samples to the CSV, creating it (with header) if needed."""
    if len(labels) != len(vectors):
        raise ValueError('Each sample needs exactly one label')

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists() or path.stat().st_size == 0

    with path.open('a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if is_new_file:
            writer.writerow(_header())
        for label, vector in zip(labels, vectors):
            if len(vector) != FEATURE_SIZE:
                raise ValueError(f'Expected {FEATURE_SIZE} features, got {len(vector)}')
            # Six decimals is well past the precision MediaPipe actually offers
            # and keeps the file readable.
            writer.writerow([label, *(f'{value:.6f}' for value in vector)])


def load_dataset(path: Path = config.SAMPLES_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Loads every sample as (X, y): features matrix and label vector."""
    if not path.exists():
        raise FileNotFoundError(f'No recordings yet at {path}. Run collect_signs.py first.')

    labels: List[str] = []
    vectors: List[List[float]] = []

    with path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f'{path} is empty')
        if header != _header():
            raise ValueError(
                f'{path} was written with a different feature layout. '
                'Delete it (or move it aside) and record again.'
            )

        for line_number, row in enumerate(reader, start=2):
            if not row:  # tolerate blank lines at the end of the file
                continue
            if len(row) != FEATURE_SIZE + 1:
                raise ValueError(f'{path}:{line_number} has {len(row)} columns, expected {FEATURE_SIZE + 1}')
            labels.append(row[0])
            vectors.append([float(value) for value in row[1:]])

    if not labels:
        raise ValueError(f'{path} has a header but no samples yet')

    return np.array(vectors, dtype=np.float64), np.array(labels)


def label_counts(path: Path = config.SAMPLES_PATH) -> Counter:
    """How many samples exist per letter, for the on-screen recording status."""
    counts: Counter = Counter()
    if not path.exists():
        return counts

    with path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            return counts
        for row in reader:
            if row:
                counts[row[0]] += 1
    return counts


def remove_last_samples(count: int, path: Path = config.SAMPLES_PATH) -> int:
    """Drops the last `count` rows, i.e. undoes a take you are not happy with.

    Returns how many rows were actually removed.
    """
    if count <= 0 or not path.exists():
        return 0

    with path.open(newline='', encoding='utf-8') as csv_file:
        rows = list(csv.reader(csv_file))

    if len(rows) <= 1:  # header only, nothing recorded
        return 0

    header, samples = rows[0], [row for row in rows[1:] if row]
    removed = min(count, len(samples))
    kept = samples[:len(samples) - removed]

    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(kept)

    return removed
