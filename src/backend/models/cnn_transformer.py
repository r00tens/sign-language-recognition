from pathlib import Path

import keras
import math
import tensorflow as tf

from src.backend.config import CHANNELS, GestureConfig, LOG_LEVEL, TrainingConfig, NUM_SELECTED
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


@keras.saving.register_keras_serializable()
class ECA(keras.api.layers.Layer):
    def __init__(self, b=1, gamma=2, **kwargs):
        super(ECA, self).__init__(**kwargs)
        self.supports_masking = True
        self.b = b
        self.gamma = gamma
        self.kernel_size = None
        self.gap = keras.api.layers.GlobalAveragePooling1D()
        self.conv = None
        self.sigmoid = keras.api.layers.Activation("sigmoid")

    def build(self, input_shape):
        channels = input_shape[-1]
        t = int(abs(math.log(channels, 2) + self.b) / self.gamma)
        self.kernel_size = t if t % 2 else t + 1

        self.conv = keras.api.layers.Conv1D(
            1, kernel_size=self.kernel_size, padding="same", use_bias=False
        )

    def call(self, inputs):
        x = self.gap(inputs)

        x = keras.api.layers.Reshape((-1, 1))(x)
        x = self.conv(x)
        x = keras.api.layers.Reshape((-1,))(x)

        x = self.sigmoid(x)

        return keras.api.layers.Multiply()([inputs, x])

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "b": self.b,
                "gamma": self.gamma,
            }
        )

        return config


@keras.saving.register_keras_serializable()
class CausalDepthwiseConv1D(keras.layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        strides=1,
        padding="valid",
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer

        self.dw_conv = keras.layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="valid",
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            depthwise_initializer=self.depthwise_initializer,
        )

    def build(self, input_shape):
        pad_amount = self.dilation_rate * (self.kernel_size - 1)
        padded_shape = list(input_shape)
        padded_shape[1] = input_shape[1] + pad_amount if input_shape[1] else None
        self.dw_conv.build(tuple(padded_shape))
        super().build(input_shape)

    def call(self, inputs):
        pad_amt = self.dilation_rate * (self.kernel_size - 1)
        # x = tf.pad(inputs, [[0, 0], [pad_amt, 0], [0, 0]])
        x = keras.ops.pad(inputs, pad_width=[[0, 0], [pad_amt, 0], [0, 0]], mode="constant")

        return self.dw_conv(x)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "depthwise_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.depthwise_initializer)
                ),
            }
        )

        return config


def multi_scale_conv_block(
    channel_size,
    kernel_sizes,
    dilation_rates,
    drop_rate=0.0,
    expand_ratio=2,
    activation="swish",
):
    def apply(inputs):
        channels_in = inputs.shape[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        # Feature expansion
        x = keras.layers.Dense(channels_expand, use_bias=True, activation=activation)(inputs)

        # Multi-scale processing
        branches = []
        for k, d in zip(kernel_sizes, dilation_rates):
            # noinspection PyCallingNonCallable
            branch = CausalDepthwiseConv1D(kernel_size=k, dilation_rate=d)(x)
            branch = keras.layers.BatchNormalization(momentum=TrainingConfig.momentum)(branch)
            # noinspection PyCallingNonCallable
            branch = ECA()(branch)
            branches.append(branch)

        # Feature fusion
        x = keras.layers.add(branches) if len(branches) > 1 else branches[0]

        # Channel projection
        x = keras.layers.Dense(channel_size, use_bias=True)(x)

        # Regularization
        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)

        # Residual connection
        if channels_in == channel_size:
            x = keras.layers.Add()([x, skip])

        return x

    return apply


def transformer_block(head_size, num_heads, expand, attn_dropout, drop_rate, activation="swish"):
    def apply(inputs):
        x = keras.api.layers.LayerNormalization()(inputs)

        # noinspection PyCallingNonCallable
        attn = keras.api.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=attn_dropout,
        )(x, x)
        attn = keras.api.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(attn)

        attn_out = keras.api.layers.Add()([inputs, attn])

        x = keras.api.layers.LayerNormalization()(attn_out)

        ff = keras.api.layers.Dense(head_size * expand, use_bias=False, activation=activation)(x)
        ff = keras.api.layers.Dense(head_size, use_bias=False)(ff)
        ff = keras.api.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(ff)

        out = keras.api.layers.Add()([attn_out, ff])

        return out

    return apply


def get_model(max_len, dim, kernel_size, num_head, expand, drop_rate, attn_dropout):
    inp = keras.api.Input((max_len, CHANNELS))

    x = keras.api.layers.Masking(mask_value=GestureConfig.pad)(inp)

    x = keras.api.layers.Dense(dim, use_bias=False)(x)
    x = keras.api.layers.BatchNormalization(momentum=TrainingConfig.momentum)(x)

    x = multi_scale_conv_block(
        channel_size=dim,
        kernel_sizes=[kernel_size, kernel_size - 2],
        dilation_rates=[1, 1],
        drop_rate=drop_rate,
    )(x)
    x = multi_scale_conv_block(
        channel_size=dim,
        kernel_sizes=[kernel_size - 4, kernel_size - 6],
        dilation_rates=[1, 1],
        drop_rate=drop_rate,
    )(x)

    x = transformer_block(
        head_size=dim,
        num_heads=num_head,
        expand=expand,
        drop_rate=drop_rate,
        attn_dropout=attn_dropout,
    )(x)

    x = multi_scale_conv_block(
        channel_size=dim,
        kernel_sizes=[kernel_size - 4, kernel_size - 6],
        dilation_rates=[1, 1],
        drop_rate=drop_rate,
    )(x)
    x = multi_scale_conv_block(
        channel_size=dim,
        kernel_sizes=[kernel_size, kernel_size - 2],
        dilation_rates=[1, 1],
        drop_rate=drop_rate,
    )(x)

    x = transformer_block(
        head_size=dim,
        num_heads=num_head,
        expand=expand,
        drop_rate=drop_rate,
        attn_dropout=attn_dropout,
    )(x)

    x = keras.api.layers.Dense(dim * 2, activation=None)(x)
    x = keras.api.layers.GlobalAveragePooling1D()(x)
    x = keras.api.layers.Dense(NUM_SELECTED)(x)

    return keras.api.Model(inp, x)


def load_keras_model(model_path: Path) -> keras.api.models.Model:
    try:
        model = keras.api.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        return model
    except Exception as ex:
        raise ValueError(f"Error loading model from {model_path}: {ex}")
