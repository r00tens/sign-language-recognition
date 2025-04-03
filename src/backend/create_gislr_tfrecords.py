import json
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.backend.config import PROJECT_ROOT, LOG_LEVEL
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)

ROWS_PER_FRAME: int = 543
CHUNK_SIZE: int = 512
N_SPLITS: int = 5
SEED: int = 42


def load_relevant_data_subset(pq_path: Union[str, Path]) -> np.ndarray:
    """
    Loads data from a Parquet file and reshapes it to a NumPy array.
    """
    pq_path = Path(pq_path)
    data = pd.read_parquet(pq_path, columns=["x", "y", "z"])
    n_frames = len(data) // ROWS_PER_FRAME
    reshaped_data = data.values.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)

    return reshaped_data


def load_label_dict(label_map_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a label mapping from a JSON file.
    """
    label_map_path = Path(label_map_path)
    with label_map_path.open("r") as file:
        return json.load(file)


def load_train_data(
    train_csv_path: Union[str, Path], label_map_path: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads training data and label mapping.
    """
    train_df = pd.read_csv(train_csv_path)
    label_dict = load_label_dict(label_map_path)

    return train_df, label_dict


def create_tf_example(
    row: pd.Series, label_dict: Dict[str, Any], base_path: Union[str, Path]
) -> tf.train.Example:
    """
    Creates a TensorFlow Example from a DataFrame row.
    """
    pq_path = Path(base_path) / row.path
    coords = load_relevant_data_subset(pq_path)

    features = {
        "coordinates": tf.train.Feature(bytes_list=tf.train.BytesList(value=[coords.tobytes()])),
        "participant_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(row.participant_id)])
        ),
        "sequence_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(row.sequence_id)])
        ),
        "sign": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_dict[row.sign])])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def process_chunk(
    chunk: pd.DataFrame,
    tfrecord_filename: Union[str, Path],
    label_dict: Dict[str, Any],
    base_path: Union[str, Path],
) -> None:
    """
    Processes a DataFrame chunk and writes it as a TFRecord file.
    """
    tfrecord_filename = Path(tfrecord_filename)
    options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)
    with tf.io.TFRecordWriter(str(tfrecord_filename), options=options) as writer:
        for _, row in tqdm(
            chunk.iterrows(), total=len(chunk), desc=f"Processing {tfrecord_filename}"
        ):
            example = create_tf_example(row, label_dict, base_path)
            writer.write(example.SerializeToString())


def split_into_chunks(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into smaller chunks.
    """
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    return [df[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


def create_stratified_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Creates stratified folds.
    """
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    logger.info(f"Creating {n_splits}-fold split for {len(df)} samples")

    for fold_idx, (train_idx, valid_idx) in enumerate(
        sgkf.split(df, y=df["sign"].values, groups=df["participant_id"])
    ):
        df.loc[valid_idx, "fold"] = fold_idx
        logger.info(
            f"Fold {fold_idx}: {len(train_idx)} training, {len(valid_idx)} validation samples"
        )

    if not (df["fold"] != -1).all():
        logger.error("Some samples were not assigned a fold!")
        raise ValueError("Some samples were not assigned a fold!")

    return df


def process_fold(
    fold: int,
    fold_df: pd.DataFrame,
    label_dict: Dict[str, Any],
    base_path: Union[str, Path],
    output_dir: Path,
) -> None:
    """
    Processes a fold by splitting data and saving TFRecord files.
    """
    logger.info(f"Processing fold {fold}...")
    chunks = split_into_chunks(fold_df, CHUNK_SIZE)
    Parallel(n_jobs=cpu_count())(
        delayed(process_chunk)(
            chunk,
            output_dir / f"fold{fold}-{i}-{len(chunk)}-{SEED}.tfrecords",
            label_dict,
            base_path,
        )
        for i, chunk in enumerate(chunks)
    )


def main() -> None:
    """
    Executes the full data processing pipeline.
    """
    train_csv_path: Path = PROJECT_ROOT / "data/large/gislr/train.csv"
    label_map_path: Path = PROJECT_ROOT / "data/large/gislr/sign_to_prediction_index_map.json"
    base_data_path: Path = PROJECT_ROOT / "data/large/gislr"
    output_dir: Path = PROJECT_ROOT / "data/large/gislr-5fold-tfrecords"

    train_df, label_dict = load_train_data(train_csv_path, label_map_path)
    train_folds = create_stratified_folds(train_df, N_SPLITS, SEED)

    logger.info(f"Sample folds:\n{train_folds.head()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(N_SPLITS):
        fold_df = train_folds[train_folds["fold"] == fold]
        process_fold(fold, fold_df, label_dict, base_data_path, output_dir)

    logger.info(f"Process completed. TFRecord dataset saved in: {output_dir}")


if __name__ == "__main__":
    main()
