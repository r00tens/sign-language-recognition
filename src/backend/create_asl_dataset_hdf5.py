import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
from mediapipe.python.solutions.hands import Hands
from tqdm import tqdm

from src.backend.config import LOG_LEVEL, PROJECT_ROOT
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.mediapipe import extract_normalized_landmarks
from src.backend.utils.miscellaneous import visualize_augmentations

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def _get_hands(min_detection_confidence: float) -> Hands:
    """
    Zwraca singleton instancji MediaPipe Hands dla procesu roboczego.

    Args:
        min_detection_confidence (float): minimalny próg pewności detekcji dłoni.

    Returns:
        Hands: skonfigurowana instancja MediaPipe Hands.
    """
    if not hasattr(_get_hands, "hands_instance"):
        _get_hands.hands_instance = Hands(  # type: ignore
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )  # type: ignore
    return _get_hands.hands_instance  # type: ignore


def _process_image_task(args: Tuple[str, str, float]) -> Tuple[Optional[list], Optional[str]]:
    """
    Przetwarza pojedynczy obraz: wczytuje plik, wykrywa punkty dłoni i normalizuje współrzędne.

    Args:
        args (Tuple[str, str, float]):
            - ścieżka do obrazu (str)
            - etykieta obrazu (str)
            - minimalna pewność detekcji dłoni (float)

    Returns:
        Tuple[Optional[list], Optional[str]]:
            - lista znormalizowanych punktów dłoni lub None,
            - etykieta obrazu lub None w przypadku błędu.
    """
    image_path, label, min_detection_confidence = args
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Nie udało się wczytać obrazu: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = _get_hands(min_detection_confidence)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        logger.error(f"Nie wykryto dłoni na obrazie: {image_path}")
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = image.shape
    landmarks = extract_normalized_landmarks(hand_landmarks, w, h)
    return landmarks, label


def process_dataset(
    dataset_dir: Path, min_detection_confidence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Przetwarza wszystkie obrazy w katalogu, wykorzystując 75% dostępnych rdzeni CPU.

    Args:
        dataset_dir (Path): katalog zawierający podfoldery z obrazami posegregowanymi według etykiet.
        min_detection_confidence (float): minimalna pewność detekcji dłoni.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - tablica znormalizowanych punktów dłoni o kształcie (N, L),
            - tablica etykiet o kształcie (N)
    """
    tasks: list[Tuple[str, str, float]] = []
    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for file_path in sorted(label_dir.iterdir()):
            if file_path.is_file():
                tasks.append((str(file_path), label, min_detection_confidence))

    cpu_count = multiprocessing.cpu_count()
    workers = max(1, int(cpu_count * 0.75))

    landmarks_list: list = []
    labels_list: list = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for landmarks, label in tqdm(
            executor.map(_process_image_task, tasks), total=len(tasks), desc="Przetwarzanie plików"
        ):
            if landmarks is not None:
                landmarks_list.append(landmarks)
                labels_list.append(label)

    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    labels_array = np.array(labels_list, dtype=str_dtype)

    return landmarks_array, labels_array


def load_hdf5_dataset(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Wczytuje dane z pliku HDF5 i dekoduje etykiety.

    Args:
        file_path (Path): ścieżka do pliku HDF5.

    Returns:
        Dict[str, np.ndarray]: słownik z kluczami:
            - 'labels': tablica etykiet,
            - 'landmarks': tablica punktów dłoni.
    """
    with h5py.File(str(file_path), "r") as h5f:
        raw_labels = h5f["labels"][:]
        decoded_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else label for label in raw_labels
        ]
        landmarks = h5f["landmarks"][:]
    return {"labels": np.array(decoded_labels), "landmarks": landmarks}


def main(_dataset_path: Path) -> None:
    """
    Główna funkcja: przetwarza zbiór i zapisuje wyniki do pliku HDF5.

    Args:
        _dataset_path (Path): ścieżka do katalogu ze zbiorami obrazów.
    """
    parent_dir = _dataset_path.parent
    _dataset_name = _dataset_path.name

    min_detection_confidence = 0.5
    logger.info(f"Zbiór danych: {_dataset_name}")

    landmarks, labels = process_dataset(_dataset_path, min_detection_confidence)

    hdf5_filename = f"{_dataset_name}-mdc-{min_detection_confidence}.h5"
    hdf5_filepath = parent_dir / hdf5_filename
    with h5py.File(hdf5_filepath, "w") as h5f:
        h5f.create_dataset("labels", data=labels)
        h5f.create_dataset("landmarks", data=landmarks)

    logger.info(f"Przetwarzanie zakończone. Zapisano dane do pliku: {hdf5_filepath}")


def display_dataset_info(_data: Dict[str, np.ndarray]) -> None:
    """
    Wyświetla w logach informacje o zbiorze: wymiary i rozkład etykiet.

    Args:
        _data (Dict[str, np.ndarray]): słownik z danymi 'labels' i 'landmarks'.
    """
    _landmarks = _data["landmarks"]
    _labels = _data["labels"]

    logger.info(f"Kształt zbioru: {_landmarks.shape}")
    unique_labels, counts = np.unique(_labels, return_counts=True)
    logger.info(f"Liczba unikalnych etykiet: {len(unique_labels)}")
    logger.info("Liczba próbek dla poszczególnych etykiet:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"Etykieta: {label}, liczba próbek: {count}")


if __name__ == "__main__":
    project_root = PROJECT_ROOT
    data_path = project_root / "data"
    large_path = data_path / "large"
    dataset_name = "madhavanair-asl-alphabet"
    dataset_path = large_path / dataset_name

    data_path.mkdir(parents=True, exist_ok=True)
    large_path.mkdir(parents=True, exist_ok=True)

    main(dataset_path)

    h5_filename = f"{dataset_name}-mdc-0.5.h5"
    h5_path = large_path / h5_filename
    try:
        data = load_hdf5_dataset(h5_path)
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas wczytywania danych: {e}")
        exit(1)

    display_dataset_info(data)

    target = "0"
    indices = np.where(data["labels"] == target)[0]
    if indices.size > 0:
        idx = indices[0]
        sample_landmarks = data["landmarks"][idx]
        logger.info(f"Wizualizacja próbki dla etykiety {target}")
        visualize_augmentations(sample_landmarks, 42)
    else:
        logger.error(f"Brak próbki dla etykiety {target}.")
