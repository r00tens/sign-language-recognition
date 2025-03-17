import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.backend.config import LETTERS, LOG_LEVEL, SEED, TRAINED_MODELS_DIR
from src.backend.create_asl_dataset_hdf5 import load_hdf5_dataset
from src.backend.data_augmentation import augment
from src.backend.models.svm import SVM
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.miscellaneous import set_seed

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def load_and_prepare_data(
    data_path: Path, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_hdf5_dataset(data_path)
    x_train = data["train"]["landmarks"]
    y_train = data["train"]["labels"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )

    logger.info(f"Kształt x_train: {x_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Kształt x_test: {x_test.shape}, y_test: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def augment_and_convert(
    x_train: np.ndarray,
    y_train: np.ndarray,
    letters: dict,
    num: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train_list = x_train.tolist()
    y_train_list = y_train.tolist()

    custom_probs = {
        "rotate_z": 0.5,
        "shear": 0.5,
        "zoom": 0.5,
        "scale": 0.5,
        "shift": 0.5,
        "jitter": 0.5,
    }

    x_train_aug, y_train_aug = augment(x_train_list, y_train_list, num=num, aug_probs=custom_probs)
    x_train_aug = np.array(x_train_aug, dtype=np.float32)
    y_train_aug = np.array(y_train_aug)
    y_train_aug = convert_labels(y_train_aug, letters)

    return x_train_aug, y_train_aug


def convert_labels(y: np.ndarray, letters: dict) -> np.ndarray:
    return np.array([letters.get(int(label), "UNKNOWN") for label in y])


def save_trained_model(svm: SVM, output_dir: Path) -> None:
    try:
        current_datetime = datetime.now().strftime("%H%M%S_%d%m%Y")

        if svm.best_params is not None:
            params_str = "-".join(f"{key}{svm.best_params[key]}" for key in svm.best_params.keys())
            model_filename = f"svm-{params_str}.pkl"
        else:
            model_filename = f"svm-default-{current_datetime}.pkl"

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / model_filename

        svm.save_model(save_path)
        logger.info(f"Model zapisany pomyślnie jako {save_path}.")
    except (OSError, pickle.PickleError) as e:
        logger.error(f"Błąd podczas zapisu modelu: {e}")


def main() -> None:
    set_seed(seed=SEED, numpy=True)

    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data"
    dataset_name = "madhavanair-asl-alphabet-mdc-0.5.h5"
    data_path = data_path / "large" / dataset_name
    x_train, x_test, y_train, y_test = load_and_prepare_data(data_path, SEED)
    letters = LETTERS
    num_augmentation = 1

    x_train_aug, y_train_aug = augment_and_convert(x_train, y_train, letters, num_augmentation)
    y_test_converted = convert_labels(y_test, letters)

    logger.info(f"Kształt x_train po augmentacji: {x_train_aug.shape}")
    logger.info(f"Kształt y_train po augmentacji: {y_train_aug.shape}")

    for label in np.unique(y_train_aug):
        logger.info(f"Etykieta: {label}, liczba próbek: {np.sum(y_train_aug == label)}")

    svm = SVM(random_state=SEED)
    svm.best_params = {
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
    }
    svm.initialize_model()

    svm.perform_cross_validation(x_train_aug, y_train_aug, cv=5)
    svm.fit(x_train_aug, y_train_aug)
    svm.evaluate(x_test, y_test_converted)

    trained_model_path = project_root / TRAINED_MODELS_DIR
    save_trained_model(svm, trained_model_path)


if __name__ == "__main__":
    main()
