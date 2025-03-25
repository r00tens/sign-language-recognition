import os
import random

import keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

from src.backend.config import LOG_LEVEL, TrainingConfig
from src.backend.data_augmentation import rotate_z, shear, zoom, scale, shift, jitter
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"Setting memory growth for GPU: {gpus}")
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logger.error(f"Error setting memory growth for GPU: {e}")
    else:
        logger.info("No GPU found.")


def set_seed(seed=42, numpy=False, tensorflow=False, krs=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if numpy:
        np.random.seed(seed)
    if tensorflow:
        tf.random.set_seed(seed)
    if krs:
        keras.api.backend.clear_session()
        keras.api.backend.set_floatx(TrainingConfig.floatx)
        keras.api.utils.set_random_seed(seed)


def visualize_hand(landmarks, title: str = "Hand visualisation"):
    points = np.array(landmarks).reshape(21, 3)

    fig, ax = plt.subplots(figsize=(8, 8))

    marker_radius = 0.025
    for i, (x, y, _) in enumerate(points):
        circle = Circle((x, y), marker_radius, facecolor="gold", edgecolor="black", zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(i), color="black", fontsize=16, ha="center", va="center", zorder=4)

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
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color="black",
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

    ax.invert_yaxis()
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", length=10, labelsize=16)
    ax.grid(True, which="both")
    fig.tight_layout()

    if not os.path.exists("../../data/augmentation-visualisation/static-gesture"):
        os.makedirs("../../data/augmentation-visualisation/static-gesture")

    plt.savefig(f"../../data/augmentation-visualisation/static-gesture/{title}.png")


def visualize_augmentations(sample_landmarks):
    augmentations = {
        "original": sample_landmarks,
        "rotate z": rotate_z(sample_landmarks.copy()),
        "shear": shear(sample_landmarks.copy()),
        "zoom": zoom(sample_landmarks.copy()),
        "scale": scale(sample_landmarks.copy()),
        "shift": shift(sample_landmarks.copy()),
        "jitter": jitter(sample_landmarks.copy()),
    }
    for name, landmarks in augmentations.items():
        logger.info(f"Visualizing {name} augmentation")
        visualize_hand(landmarks, title=name)
