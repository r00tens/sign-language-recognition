from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from src.backend.config import LOG_LEVEL
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.mediapipe import extract_normalized_landmarks
from src.backend.utils.miscellaneous import visualize_augmentations

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def process_image(image_path: Path, hands) -> Optional[list]:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Nie udało się wczytać obrazu: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image.shape
        landmarks = extract_normalized_landmarks(hand_landmarks, w, h)

        return landmarks
    else:
        logger.error(f"Nie wykryto dłoni na obrazie: {image_path}")
        return None


def process_dataset(dataset_dir: Path, set_type: str, hands) -> tuple[np.ndarray, np.ndarray]:
    base_path = dataset_dir / set_type
    landmarks_list = []
    labels_list = []
    for label_dir in tqdm(
        sorted(base_path.iterdir()),
        desc=f"Przetwarzanie etykiet dla {set_type}",
        position=0,
        leave=True,
    ):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        for file_path in tqdm(
            sorted(label_dir.iterdir()), desc=f"Etykieta {label}", position=0, leave=True
        ):
            if file_path.is_file():
                landmarks = process_image(file_path, hands)

                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    labels_list.append(label)

    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    labels_array = np.array(labels_list, dtype=str_dtype)

    return landmarks_array, labels_array


def load_hdf5_dataset(file_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(str(file_path), "r") as h5f:
        for set_type in h5f.keys():
            raw_labels = h5f[f"{set_type}/labels"][:]
            decoded_labels = [
                label.decode("utf-8") if isinstance(label, bytes) else label
                for label in raw_labels
            ]
            dataset[set_type] = {
                "labels": np.array(decoded_labels),
                "landmarks": h5f[f"{set_type}/landmarks"][:],
            }

    return dataset


def main(_dataset_path: Path) -> None:
    parent_dir = _dataset_path.parent
    _dataset_name = _dataset_path.name

    min_detection_confidence = 0.5
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    logger.info(f"Zbiór danych: {_dataset_name}")
    logger.info("Przetwarzanie zbioru treningowego")
    _train_landmarks, _train_labels = process_dataset(_dataset_path, "train", hands)
    logger.info("Przetwarzanie zbioru testowego")
    _test_landmarks, _test_labels = process_dataset(_dataset_path, "test", hands)

    hands.close()

    hdf5_filename = f"{_dataset_name}-mdc-{min_detection_confidence}.h5"
    hdf5_filepath = parent_dir / hdf5_filename

    with h5py.File(hdf5_filepath, "w") as h5f:
        train_grp = h5f.create_group("train")
        train_grp.create_dataset("labels", data=_train_labels)
        train_grp.create_dataset("landmarks", data=_train_landmarks)

        test_grp = h5f.create_group("test")
        test_grp.create_dataset("labels", data=_test_labels)
        test_grp.create_dataset("landmarks", data=_test_landmarks)

    logger.info(f"Przetwarzanie zakończone. Zapisano dane do pliku: {hdf5_filepath}")


def display_dataset_info(_data: Dict[str, Dict[str, np.ndarray]]):
    _train_landmarks = _data["train"]["landmarks"]
    _train_labels = _data["train"]["labels"]
    _test_landmarks = _data["test"]["landmarks"]
    _test_labels = _data["test"]["labels"]

    logger.info(f"Kształt zbioru treningowego: {_train_landmarks.shape}")
    logger.info(f"Kształt zbioru testowego: {_test_landmarks.shape}")

    _unique_train_labels, _unique_train_labels_counts = np.unique(
        _train_labels, return_counts=True
    )
    _unique_test_labels, _unique_test_labels_counts = np.unique(_test_labels, return_counts=True)

    logger.info(f"Liczba unikalnych etykiet w zbiorze treningowym: {len(_unique_train_labels)}")
    logger.info(f"Liczba unikalnych etykiet w zbiorze testowym: {len(_unique_test_labels)}")

    logger.info("Liczba próbek dla poszczególnych etykiet w zbiorze treningowym:")
    for label, count in zip(_unique_train_labels, _unique_train_labels_counts):
        logger.info(f"Etykieta: {label}, liczba próbek: {count}")
    logger.info("Liczba próbek dla poszczególnych etykiet w zbiorze testowym:")
    for label, count in zip(_unique_test_labels, _unique_test_labels_counts):
        logger.info(f"Etykieta: {label}, liczba próbek: {count}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data"
    large_path = data_path / "large"
    dataset_name = "madhavanair-asl-alphabet"
    dataset_path = data_path / "large" / dataset_name

    if not data_path.exists():
        logger.warning(f"Folder 'data' nie istnieje, tworzę: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)

    if not large_path.exists():
        logger.warning(f"Folder 'large' nie istnieje, tworzę: {large_path}")
        large_path.mkdir(parents=True, exist_ok=True)

    main(dataset_path)

    dataset_name = "madhavanair-asl-alphabet-mdc-0.5.h5"
    dataset_path = data_path / "large" / dataset_name
    try:
        data = load_hdf5_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas wczytywania danych: {e}")
        exit(1)

    display_dataset_info(data)

    train_landmarks = data["train"]["landmarks"]
    train_labels = data["train"]["labels"]
    target = "5"
    indices = np.where(train_labels == target)[0]
    if indices.size > 0:
        idx = indices[0]
        sample_landmarks = train_landmarks[idx]

        logger.info(f"Wizualizacja próbki dla etykiety {target}")

        visualize_augmentations(sample_landmarks)
    else:
        logger.error(f"Brak próbki dla etykiety {target} w zbiorze treningowym.")
