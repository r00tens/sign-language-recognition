from pathlib import Path
from typing import Tuple

import keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.backend.config import (
    LOG_LEVEL,
    LANDMARKS,
    GestureConfig,
    TrainingConfig,
    ALLOWED_LABELS,
    NUM_SELECTED,
    TRAINED_MODELS_DIR,
    CHANNELS,
    MODEL_CNN_TRANSFORMER,
    SEED,
    PROJECT_ROOT,
)
from src.backend.data_augmentation_tf import tf_nan_mean, tf_nan_std, augment_fn
from src.backend.models.cnn_transformer import (
    get_model,
    load_keras_model,
)
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.callbacks import LearningRateTracker
from src.backend.utils.training_setup import (
    set_seed,
    configure_gpu_memory_growth,
    set_mixed_precision_policy,
    get_strategy,
)

logger = AppLogger(__name__, level=LOG_LEVEL)

configure_gpu_memory_growth()


class Preprocess(keras.api.layers.Layer):
    def __init__(
        self,
        max_len=GestureConfig.max_len,
        point_landmarks=LANDMARKS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    @staticmethod
    def ensure_batch_dim(inputs: tf.Tensor) -> tf.Tensor:
        return tf.cond(
            tf.equal(tf.rank(inputs), 3),
            lambda: tf.expand_dims(inputs, axis=0),
            lambda: inputs,
        )

    def compute_normalized_features(self, x: tf.Tensor) -> tf.Tensor:
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)

        x = tf.gather(x, self.point_landmarks, axis=2)  # shape: N, T, P, C
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std

        return x

    def crop_and_extract_xy(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.max_len is not None:
            x = x[:, : self.max_len]

        length = tf.shape(x)[1]
        x = x[..., :2]

        return x, length

    @staticmethod
    def compute_temporal_derivatives(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dx = tf.cond(
            tf.greater(tf.shape(x)[1], 1),
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )

        dx2 = tf.cond(
            tf.greater(tf.shape(x)[1], 2),
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )

        return dx, dx2

    def reshape_and_concat_features(
        self, x: tf.Tensor, dx: tf.Tensor, dx2: tf.Tensor, length: tf.Tensor
    ) -> tf.Tensor:
        num_landmarks = len(self.point_landmarks)

        x_reshaped = tf.reshape(x, (-1, length, 2 * num_landmarks))
        dx_reshaped = tf.reshape(dx, (-1, length, 2 * num_landmarks))
        dx2_reshaped = tf.reshape(dx2, (-1, length, 2 * num_landmarks))

        return tf.concat([x_reshaped, dx_reshaped, dx2_reshaped], axis=-1)

    def call(self, inputs):
        x = self.ensure_batch_dim(inputs)
        x = self.compute_normalized_features(x)

        x, length = self.crop_and_extract_xy(x)

        dx, dx2 = self.compute_temporal_derivatives(x)

        features = self.reshape_and_concat_features(x, dx, dx2, length)
        features = tf.where(
            tf.math.is_nan(features), tf.constant(0.0, dtype=features.dtype), features
        )

        return features


def filter_nans_tf(x, ref_point=LANDMARKS):
    mask = tf.math.logical_not(
        tf.reduce_all(tf.math.is_nan(tf.gather(x, ref_point, axis=1)), axis=[-2, -1])
    )
    x = tf.boolean_mask(x, mask, axis=0)

    return x


def preprocess(x, augment=False, max_len=GestureConfig.max_len):
    coord = x["coordinates"]
    coord = filter_nans_tf(coord)

    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, GestureConfig.rows_per_frame, 3))

    # noinspection PyCallingNonCallable
    proc_coord = tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(ALLOWED_LABELS, dtype=tf.int64),
            values=tf.range(NUM_SELECTED, dtype=tf.int64),
        ),
        default_value=-1,
    )
    new_label = table.lookup(x["sign"])
    one_hot_label = tf.one_hot(new_label, depth=NUM_SELECTED)

    return proc_coord, one_hot_label


def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(
        record_bytes,
        {
            "coordinates": tf.io.FixedLenFeature([], tf.string),
            "sign": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    out = {
        "coordinates": tf.reshape(
            tf.io.decode_raw(features["coordinates"], tf.float32),
            (-1, GestureConfig.rows_per_frame, 3),
        ),
        "sign": features["sign"],
    }

    return out


def get_tfrec_dataset(
    tfrecords,
    batch_size,
    max_len,
    drop_remainder=False,
    augment=False,
    shuffle=0,
    repeat=False,
):
    allowed_labels_tensor = tf.constant(ALLOWED_LABELS, dtype=tf.int64)

    ds = tf.data.TFRecordDataset(
        tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
    )
    ds = ds.map(decode_tfrec, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.reduce_any(tf.equal(x["sign"], allowed_labels_tensor)))

    # unique_signs = set()
    # for x in ds:
    #     unique_signs.add(x["sign"].numpy())
    # logger.info(f"Unique signs: {unique_signs.__len__()}")

    ds = ds.map(
        lambda x: preprocess(x, augment=augment, max_len=max_len),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

    if batch_size:
        ds = ds.padded_batch(
            batch_size,
            padding_values=GestureConfig.pad,
            padded_shapes=([max_len, CHANNELS], [NUM_SELECTED]),
            drop_remainder=drop_remainder,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def count_data_items(tfrecords):
    ds = get_tfrec_dataset(
        tfrecords,
        batch_size=1,
        max_len=GestureConfig.max_len,
        augment=False,
        shuffle=0,
        repeat=False,
    )
    count = 0

    for batch in ds:
        count += batch[0].shape[0]

    return count


def plot_accuracy(history):
    acc = history.history.get("categorical_accuracy")
    val_acc = history.history.get("val_categorical_accuracy")
    if not acc:
        return
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, acc, "-", label="Training Accuracy")
    if val_acc:
        ax.plot(epochs, val_acc, "-", label="Validation Accuracy")

    max_train_idx = np.argmax(acc)
    max_train_acc = acc[max_train_idx]
    ax.scatter(max_train_idx + 1, max_train_acc, s=30, zorder=5)
    ax.annotate(
        f"Max: {max_train_acc:.2f}",
        (max_train_idx + 1, max_train_acc),
        textcoords="offset points",
        xytext=(0, -12),
        ha="center",
    )

    if val_acc:
        max_val_idx = np.argmax(val_acc)
        max_val_acc = val_acc[max_val_idx]
        ax.scatter(max_val_idx + 1, max_val_acc, s=30, zorder=5)
        ax.annotate(
            f"Max: {max_val_acc:.2f}",
            (max_val_idx + 1, max_val_acc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    plt.savefig(PROJECT_ROOT / "trained-models/training-accuracy.png")
    plt.close(fig)


def plot_loss(history):
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, loss, "-", label="Training Loss")
    if val_loss:
        ax.plot(epochs, val_loss, "-", label="Validation Loss")

    min_train_idx = np.argmin(loss)
    min_train_loss = loss[min_train_idx]
    ax.scatter(min_train_idx + 1, min_train_loss, s=30, zorder=5)
    ax.annotate(
        f"Min: {min_train_loss:.2f}",
        (min_train_idx + 1, min_train_loss),
        textcoords="offset points",
        xytext=(0, -12),
        ha="center",
    )

    if val_loss:
        min_val_idx = np.argmin(val_loss)
        min_val_loss = val_loss[min_val_idx]
        ax.scatter(min_val_idx + 1, min_val_loss, s=30, zorder=5)
        ax.annotate(
            f"Min: {min_val_loss:.2f}",
            (min_val_idx + 1, min_val_loss),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(PROJECT_ROOT / "trained-models/training-loss.png")
    plt.close(fig)


def plot_history(history):
    plot_accuracy(history)
    plot_loss(history)


def plot_lr_vs_loss(history):
    num_epochs = len(history.history["loss"])
    lrs = [1e-6 * 10 ** (epoch / 40) for epoch in range(num_epochs)]
    losses = history.history["loss"]

    plt.figure(figsize=(8, 6))
    plt.plot(lrs, losses, marker="o")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs Loss")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.xlim(1e-6, 1e-1)

    ticks = np.logspace(np.log10(1e-6), np.log10(1e-1), num=15)

    ax = plt.gca()
    ax.set_xticks(ticks)

    ax.set_xticklabels([f"{tick:.0e}" for tick in ticks], rotation=45)

    plt.savefig(PROJECT_ROOT / "trained-models/lr-vs-loss.png")


def train_fold(fold, strategy, train_files, valid_files=None, summary=True):
    if fold != "all":
        train_ds = get_tfrec_dataset(
            train_files,
            batch_size=TrainingConfig.batch_size,
            max_len=GestureConfig.max_len,
            drop_remainder=True,
            augment=True,
            repeat=True,
            shuffle=32768,
        )
        valid_ds = get_tfrec_dataset(
            valid_files,
            batch_size=TrainingConfig.batch_size,
            max_len=GestureConfig.max_len,
            drop_remainder=False,
            repeat=False,
            shuffle=0,
        )
    else:
        train_ds = get_tfrec_dataset(
            train_files,
            batch_size=TrainingConfig.batch_size,
            max_len=GestureConfig.max_len,
            drop_remainder=False,
            augment=True,
            repeat=True,
            shuffle=32768,
        )
        valid_ds = None
        valid_files = []

    num_train = count_data_items(train_files)
    num_valid = count_data_items(valid_files)
    steps_per_epoch = num_train // TrainingConfig.batch_size

    with strategy.scope():
        model = get_model(
            max_len=GestureConfig.max_len,
            dim=TrainingConfig.dim,
            kernel_size=TrainingConfig.kernel_size,
            num_head=TrainingConfig.num_heads,
            expand=TrainingConfig.expand,
            drop_rate=TrainingConfig.drop_rate,
            attn_dropout=TrainingConfig.attn_dropout,
        )

        lr_schedule = keras.api.optimizers.schedules.CosineDecay(
            initial_learning_rate=TrainingConfig.learning_rate,
            decay_steps=steps_per_epoch * TrainingConfig.epochs,
            alpha=TrainingConfig.min_lr / TrainingConfig.learning_rate,
        )

        optimizer = keras.api.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=TrainingConfig.weight_decay,
            clipnorm=TrainingConfig.clipnorm,
            clipvalue=TrainingConfig.clipvalue,
            use_ema=True,
            ema_momentum=0.995,
            ema_overwrite_frequency=None,
        )

        model.compile(
            optimizer=optimizer,
            loss=[
                keras.api.losses.CategoricalCrossentropy(
                    from_logits=True, label_smoothing=TrainingConfig.label_smoothing
                )
            ],
            metrics=[
                keras.api.metrics.CategoricalAccuracy(),
            ],
            steps_per_execution=steps_per_epoch,
        )

    if summary:
        model.summary()

        logger.info(f"Train dataset: {train_ds}")
        logger.info(f"Valid dataset: {valid_ds}")

        for x in train_ds.take(1):
            logger.info(f"Coordinates shape: {x[0].shape}")
            logger.info(f"Labels shape: {x[1].shape}")

    logger.info(f"FOLD: {fold}")
    logger.info(f"Train: {num_train} samples")
    logger.info(f"Valid: {num_valid} samples")

    # fmt: off
    callbacks = [
        keras.api.callbacks.CSVLogger(PROJECT_ROOT / f"trained-models/training-logs.csv"),
        keras.api.callbacks.SwapEMAWeights(swap_on_epoch=True),
    ]
    # fmt: on

    checkpoint_full = keras.api.callbacks.ModelCheckpoint(
        filepath=PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-best.keras",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
    )

    checkpoint_weights = keras.api.callbacks.ModelCheckpoint(
        filepath=PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-best.weights.h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    if fold != "all":
        callbacks.append(checkpoint_full)
        callbacks.append(checkpoint_weights)
        callbacks.append(
            LearningRateTracker(PROJECT_ROOT / "trained-models/learning-rate-plot.png")
        )

    history = model.fit(
        train_ds,
        epochs=TrainingConfig.epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=TrainingConfig.verbose,
        callbacks=callbacks,
        validation_data=valid_ds,
        validation_steps=-(num_valid // -TrainingConfig.batch_size),
    )

    plot_history(history)

    model.save(PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-final.keras")
    model.save_weights(
        PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-final.weights.h5"
    )

    if fold != "all":
        try:
            model.load_weights(
                PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-best.weights.h5"
            )
            logger.info(f"Model loaded with best weights")
        except Exception as e:
            logger.error(f"Failed to load best weights: {e}")

        logger.info(f"Evaluating best model")
        cv = model.evaluate(valid_ds, verbose=2, steps=-(num_valid // -TrainingConfig.batch_size))

        try:
            model.load_weights(
                PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-final.weights.h5"
            )
            logger.info(f"Model loaded with final weights")
        except Exception as e:
            logger.error(f"Failed to load final weights: {e}")

        logger.info(f"Evaluating final model")
        cv_final = model.evaluate(
            valid_ds, verbose=2, steps=-(num_valid // -TrainingConfig.batch_size)
        )
    else:
        cv = None

        try:
            model.load_weights(
                PROJECT_ROOT / f"trained-models/{TrainingConfig.model_name}-final.weights.h5"
            )
            logger.info(f"Model loaded with final weights")
        except Exception as e:
            logger.error(f"Failed to load final weights: {e}")

        logger.info(f"Evaluating final model")
        cv_final = model.evaluate(
            valid_ds, verbose=2, steps=-(num_valid // -TrainingConfig.batch_size)
        )

    return model, cv, cv_final, history


def train_folds(folds, strategy, summary=True):
    data_dir = PROJECT_ROOT / "data/large/gislr-5fold-tfrecords"
    train_filenames = list(data_dir.glob("*.tfrecords"))

    for fold in folds:
        if fold != "all":
            all_files = train_filenames
            train_files = [x for x in all_files if f"fold{fold}" not in x.name]
            valid_files = [x for x in all_files if f"fold{fold}" in x.name]
        else:
            train_files = train_filenames
            valid_files = None

        train_fold(fold, strategy, train_files, valid_files, summary)


class TFLiteModel(tf.Module):
    def __init__(self, keras_model):
        super(TFLiteModel, self).__init__()
        self.prepare_inputs = Preprocess()
        self.keras_model = keras_model

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")]
    )
    def __call__(self, inputs):
        # noinspection PyCallingNonCallable
        x = self.prepare_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = self.keras_model(x)
        return {"outputs": outputs}


def convert_to_tflite(keras_model, output_dir: Path):
    try:
        tflite_model_wrapper = TFLiteModel(keras_model)
        concrete_func = tflite_model_wrapper.__call__.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        with open(output_dir / MODEL_CNN_TRANSFORMER, "wb") as f:
            f.write(tflite_model)

        logger.info(f"Model converted to TFLite: {output_dir / MODEL_CNN_TRANSFORMER}")
    except Exception as e:
        raise ValueError(f"Failed to convert model to TFLite: {e}")


def analyze_tflite_model(model_path: Path, gpu_compatibility=True):
    try:
        tf.lite.experimental.Analyzer.analyze(str(model_path), gpu_compatibility=gpu_compatibility)
    except Exception as e:
        raise ValueError(f"Failed to analyze TFLite model: {e}")


def main():
    keras.api.backend.clear_session()
    set_seed(SEED, numpy=True, tensorflow=True, krs=True)
    set_mixed_precision_policy(TrainingConfig.precision)
    tf.config.optimizer.set_jit("autoclustering")

    strategy, replicas = get_strategy(TrainingConfig.device)

    train_folds([0], strategy)

    project_root = Path(__file__).resolve().parents[3]

    model_name = "cnn-transformer-best"
    model_file_type = "keras"
    model_path = project_root / TRAINED_MODELS_DIR / f"{model_name}.{model_file_type}"

    try:
        model = load_keras_model(model_path)
        convert_to_tflite(model, project_root / TRAINED_MODELS_DIR)
    except Exception as e:
        logger.error(f"{e}")
        exit(1)

    # model_tflite_path = project_root / TRAINED_MODELS_DIR / MODEL_CNN_TRANSFORMER

    # analyze_tflite_model(model_tflite_path)

    # interpreter = tf.lite.Interpreter(str(model_tflite_path))
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # logger.info(f"Input details: {input_details}")
    # logger.info(f"Output details: {output_details}")


if __name__ == "__main__":
    main()
