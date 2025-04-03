import os
import random

import keras
import numpy as np
import tensorflow as tf

from src.backend.config import LOG_LEVEL
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
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)

        if numpy:
            np.random.seed(seed)
            logger.info(f"NumPy seed set to {seed}")
        if tensorflow:
            tf.random.set_seed(seed)
            logger.info(f"TensorFlow seed set to {seed}")
        if krs:
            keras.api.utils.set_random_seed(seed)
            logger.info(f"Keras seed set to {seed}")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")


def set_mixed_precision_policy(precision):
    if precision == "float16":
        policy = keras.mixed_precision.Policy("mixed_float16")
        keras.mixed_precision.set_global_policy(policy)

        logger.info("Using mixed precision float16")
    elif precision == "bfloat16":
        policy = keras.mixed_precision.Policy("mixed_bfloat16")
        keras.mixed_precision.set_global_policy(policy)

        logger.info("Using mixed precision bfloat16")
    elif precision == "float32":
        policy = keras.mixed_precision.Policy("float32")
        keras.mixed_precision.set_global_policy(policy)

        logger.info("Using float32")
    else:
        raise ValueError(f"Unsupported float type: {precision}")


def get_strategy(device):
    if "TPU" in device:
        tpu_resolver = "local" if device == "TPU-VM" else None

        logger.info(f"Connecting to TPU: {tpu_resolver}")

        tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu_resolver)
        strategy = tf.distribute.TPUStrategy(tpu_cluster)
        num_gpu = 0
    elif device in ("GPU", "CPU"):
        physical_gpus = tf.config.experimental.list_physical_devices("GPU")
        num_gpu = len(physical_gpus)

        if num_gpu > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using {num_gpu} GPUs")
        elif num_gpu == 1:
            strategy = tf.distribute.get_strategy()
            logger.info(f"Using single GPU: {physical_gpus[0].name}")
        else:
            strategy = tf.distribute.get_strategy()
            logger.info(f"Using CPU")
    else:
        num_gpu = 0
        strategy = tf.distribute.get_strategy()
        logger.error("Device not recognized, defaulting to CPU")

    if device == "GPU":
        logger.info(f"Num GPUs Available: {num_gpu}")

    replicas = strategy.num_replicas_in_sync
    logger.info(f"Number of replicas: {replicas}")

    return strategy, replicas
