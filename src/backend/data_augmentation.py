import random

import numpy as np
from tqdm import tqdm


def rotate_z(landmarks):
    angle = np.random.uniform(-30, 30)
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return _transform_landmarks(landmarks, lambda shifted: np.dot(shifted, rotation_matrix.T))


def shear(landmarks):
    shear_value = np.random.uniform(-0.15, 0.15)

    if np.random.rand() < 0.5:
        shear_x = shear_value
        shear_y = 0.0
    else:
        shear_x = 0.0
        shear_y = shear_value

    shear_matrix = np.array([[1, shear_x], [shear_y, 1]])

    return _transform_landmarks(landmarks, lambda shifted: np.dot(shifted, shear_matrix.T))


def zoom(landmarks):
    factor = np.random.uniform(0.8, 1.2)
    landmarks = np.asarray(landmarks).reshape(21, 3)
    center = np.mean(landmarks[:, :2], axis=0)
    landmarks[:, :2] = (landmarks[:, :2] - center) * factor + center

    return landmarks.flatten().tolist()


def scale(landmarks):
    factor = np.random.uniform(0.8, 1.2)
    landmarks = np.asarray(landmarks).reshape(21, 3)
    landmarks = landmarks * factor

    return landmarks.flatten().tolist()


def shift(landmarks):
    landmarks = np.asarray(landmarks).reshape(21, 3)
    dim = landmarks.shape[-1]

    if dim == 3:
        x_shift = np.random.uniform(-0.1, 0.1)
        y_shift = np.random.uniform(-0.1, 0.1)
        z_shift = np.random.uniform(-0.1, 0.1)

        landmarks[:, 0] += x_shift
        landmarks[:, 1] += y_shift
        landmarks[:, 2] += z_shift
    else:
        raise ValueError("Nieobsługiwany wymiar landmarków. Oczekiwano 3.")

    return landmarks.flatten().tolist()


def jitter(landmarks, noise_level=0.01):
    landmarks = np.asarray(landmarks).reshape(21, 3)
    noise = np.random.normal(0, noise_level, landmarks.shape)
    landmarks += noise
    return landmarks.flatten().tolist()


def apply_augmentations(data, aug_probs=None):
    aug_functions = [rotate_z, shear, zoom, scale, shift, jitter]

    if aug_probs is None:
        aug_probs = {func.__name__: 0.5 for func in aug_functions}

    random.shuffle(aug_functions)
    applied = False

    for fun in aug_functions:
        prob = aug_probs.get(fun.__name__, 0.5)
        if np.random.rand() < prob:
            data = fun(data)
            applied = True

    if not applied:
        data = random.choice(aug_functions)(data)

    return data


def augment(x, y, num=None, aug_probs=None):
    x_aug = x.copy()
    y_aug = y.copy()

    if num is None:
        for i in tqdm(range(len(y)), ncols=100):
            num_aug = np.random.choice([1, 2, 3])
            for n in range(num_aug):
                x_aug.append(apply_augmentations(x[i].copy(), aug_probs=aug_probs))
                y_aug.append(y[i])
    elif num > 0:
        for i in tqdm(range(len(y)), ncols=100):
            for n in range(num):
                x_aug.append(apply_augmentations(x[i].copy(), aug_probs=aug_probs))
                y_aug.append(y[i])

    print(apply_augmentations.__doc__)

    return x_aug, y_aug


def _transform_landmarks(landmarks, transform_fn):
    landmarks = np.asarray(landmarks).reshape(21, 3)
    center = np.mean(landmarks[:, :2], axis=0)
    shifted = landmarks[:, :2] - center
    transformed = transform_fn(shifted)
    landmarks[:, :2] = transformed + center

    return landmarks.flatten().tolist()
