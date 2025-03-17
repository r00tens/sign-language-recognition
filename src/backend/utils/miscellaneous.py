import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.backend.config import LOG_LEVEL
from src.backend.data_augmentation import rotate_z, shear, zoom, scale, shift, jitter
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def set_seed(seed=42, numpy=False, tensorflow=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if numpy:
        np.random.seed(seed)
    if tensorflow:
        tf.random.set_seed(seed)


def visualize_hand(landmarks, title: str = "Wizualizacja dłoni"):
    points = np.array(landmarks).reshape(21, 3)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=250,
        zorder=3,
        color="gold",
        edgecolors="black",
    )

    for i, (x, y, _) in enumerate(points):
        plt.text(x, y, str(i), color="black", fontsize=10, ha="center", va="center", zorder=4)

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for i, j in connections:
        plt.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color="black",
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

    plt.gca().invert_yaxis()
    plt.title(title, fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_augmentations(sample_landmarks):
    augmentations = {
        "Oryginał": sample_landmarks,
        "Rotate Z": rotate_z(sample_landmarks.copy()),
        "Shear": shear(sample_landmarks.copy()),
        "Zoom": zoom(sample_landmarks.copy()),
        "Scale": scale(sample_landmarks.copy()),
        "Shift": shift(sample_landmarks.copy()),
        "Jitter": jitter(sample_landmarks.copy()),
    }
    for name, landmarks in augmentations.items():
        logger.info(f"Wizualizacja dla: {name}")
        visualize_hand(landmarks, title=name)
