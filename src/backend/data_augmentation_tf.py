from typing import Tuple, Union, Sequence

import tensorflow as tf

from src.backend.config import (
    LHAND,
    RHAND,
    LLIP,
    RLIP,
    LPOSE,
    RPOSE,
    LEYE,
    REYE,
    LNOSE,
    RNOSE,
    GestureConfig,
)


def swap_keypoints(tensor: tf.Tensor, idx1, idx2, axis: int = 0) -> tf.Tensor:
    kp1 = tf.gather(tensor, idx1, axis=axis)
    kp2 = tf.gather(tensor, idx2, axis=axis)

    indices1 = tf.reshape(tf.constant(idx1), (-1, 1))
    indices2 = tf.reshape(tf.constant(idx2), (-1, 1))

    tensor = tf.tensor_scatter_nd_update(tensor, indices1, kp2)
    tensor = tf.tensor_scatter_nd_update(tensor, indices2, kp1)

    return tensor


def flip_lr(x: tf.Tensor) -> tf.Tensor:
    x_channel, y_channel, z_channel = tf.unstack(x, axis=-1)
    flipped_x_channel = 1 - x_channel

    flipped = tf.stack([flipped_x_channel, y_channel, z_channel], axis=-1)
    flipped = tf.transpose(flipped, perm=[1, 0, 2])
    flipped = swap_keypoints(flipped, LHAND, RHAND)
    flipped = swap_keypoints(flipped, LLIP, RLIP)
    flipped = swap_keypoints(flipped, LPOSE, RPOSE)
    flipped = swap_keypoints(flipped, LEYE, REYE)
    flipped = swap_keypoints(flipped, LNOSE, RNOSE)
    flipped = tf.transpose(flipped, perm=[1, 0, 2])

    return flipped


def resize_with_interpolation(
    x: tf.Tensor, target_len: tf.Tensor, method: str = "random"
) -> tf.Tensor:
    target_len = tf.maximum(1, target_len)
    new_size = (target_len, tf.shape(x)[1])

    if method == "random":
        rand_val = tf.random.uniform(())

        def resize_bilinear():
            return tf.image.resize(x, new_size, method="bilinear")

        def resize_bicubic():
            return tf.image.resize(x, new_size, method="bicubic")

        def resize_nearest():
            return tf.image.resize(x, new_size, method="nearest")

        resized = tf.cond(
            tf.less(rand_val, 0.33),
            resize_bilinear,
            lambda: tf.cond(tf.less(rand_val, 0.83), resize_bicubic, resize_nearest),
        )
        return resized
    else:
        return tf.image.resize(x, new_size, method=method)


def resample(x, rate=(0.8, 1.2)):
    rate_val = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate_val * tf.cast(length, tf.float32), tf.int32)

    return resize_with_interpolation(x, new_size)


def apply_scaling(xyz: tf.Tensor, scale_range: Tuple[float, float]) -> tf.Tensor:
    scale_factor = tf.random.uniform((), *scale_range)
    return scale_factor * xyz


def apply_shear(
    xyz: tf.Tensor, shear_range: Tuple[float, float], center: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    shear_val = tf.random.uniform((), *shear_range)

    if tf.random.uniform(()) < 0.5:
        shear_x, shear_y = 0.0, shear_val
    else:
        shear_x, shear_y = shear_val, 0.0

    shear_matrix = tf.convert_to_tensor([[1.0, shear_x], [shear_y, 1.0]], dtype=xyz.dtype)
    xy_transformed = tf.matmul(xy, shear_matrix)
    new_xyz = tf.concat([xy_transformed, z], axis=-1)
    new_center = center + tf.stack([shear_y, shear_x])

    return new_xyz, new_center


def apply_rotation(
    xyz: tf.Tensor, degree_range: Tuple[float, float], center: tf.Tensor
) -> tf.Tensor:
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    xy_centered = xy - center

    degree_val = tf.random.uniform((), *degree_range)
    radian = degree_val * tf.constant(tf.experimental.numpy.pi / 180, dtype=xyz.dtype)
    cos_val = tf.math.cos(radian)
    sin_val = tf.math.sin(radian)

    rotation_matrix = tf.convert_to_tensor(
        [[cos_val, sin_val], [-sin_val, cos_val]], dtype=xyz.dtype
    )
    xy_rotated = tf.matmul(xy_centered, rotation_matrix) + center

    return tf.concat([xy_rotated, z], axis=-1)


def apply_shift(xyz: tf.Tensor, shift_range: Tuple[float, float]) -> tf.Tensor:
    shift_val = tf.random.uniform((), *shift_range)

    return xyz + shift_val


def spatial_random_affine(
    xyz: tf.Tensor,
    scale: Tuple[float, float] = (0.8, 1.2),
    shear: Tuple[float, float] = (-0.15, 0.15),
    shift: Tuple[float, float] = (-0.1, 0.1),
    degree: Tuple[float, float] = (-30, 30),
) -> tf.Tensor:
    center = tf.constant([0.5, 0.5], dtype=xyz.dtype)

    if scale is not None:
        xyz = apply_scaling(xyz, scale)

    if shear is not None:
        xyz, center = apply_shear(xyz, shear, center)

    if degree is not None:
        xyz = apply_rotation(xyz, degree, center)

    if shift is not None:
        xyz = apply_shift(xyz, shift)

    return xyz


def temporal_crop(x: tf.Tensor, length: int = GestureConfig.max_len) -> tf.Tensor:
    seq_length = tf.shape(x)[0]

    def crop():
        max_offset = tf.maximum(seq_length - length, 1)
        offset = tf.random.uniform((), 0, max_offset, dtype=tf.int32)

        return x[offset : offset + length]

    return tf.cond(seq_length >= length, crop, lambda: x)


def temporal_mask(
    x: tf.Tensor, size: tuple = (0.2, 0.4), mask_value: float = float("nan")
) -> tf.Tensor:
    seq_length = tf.shape(x)[0]

    mask_ratio = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(seq_length, tf.float32) * mask_ratio, tf.int32)
    max_offset = tf.maximum(seq_length - mask_size, 1)
    mask_offset = tf.random.uniform((), 0, max_offset, dtype=tf.int32)
    mask_indices = tf.reshape(tf.range(mask_offset, mask_offset + mask_size), (-1, 1))
    mask_tensor = tf.fill([mask_size, GestureConfig.rows_per_frame, 3], mask_value)

    return tf.tensor_scatter_nd_update(x, mask_indices, mask_tensor)


def spatial_mask(
    x: tf.Tensor, size: tuple = (0.2, 0.4), mask_value: float = float("nan")
) -> tf.Tensor:
    mask_offset_x = tf.random.uniform(())
    mask_offset_y = tf.random.uniform(())

    mask_size = tf.random.uniform((), *size)
    mask_x = tf.logical_and(x[..., 0] > mask_offset_x, x[..., 0] < mask_offset_x + mask_size)
    mask_y = tf.logical_and(x[..., 1] > mask_offset_y, x[..., 1] < mask_offset_y + mask_size)
    mask = tf.logical_and(mask_x, mask_y)
    mask_expanded = tf.expand_dims(mask, axis=-1)

    return tf.where(mask_expanded, mask_value, x)


def augment_fn(x: tf.Tensor, max_len: int, always: bool = False) -> tf.Tensor:
    if always or tf.random.uniform(()) < 0.8:
        x = resample(x, (0.5, 1.5))

    if always or tf.random.uniform(()) < 0.5:
        x = flip_lr(x)

    if max_len is not None:
        x = temporal_crop(x, max_len)

    if always or tf.random.uniform(()) < 0.75:
        x = spatial_random_affine(x)

    if always or tf.random.uniform(()) < 0.5:
        x = temporal_mask(x)

    if always or tf.random.uniform(()) < 0.5:
        x = spatial_mask(x)

    return x


def tf_nan_mean(
    x: tf.Tensor, axis: Union[int, Sequence[int]] = 0, keepdims: bool = False
) -> tf.Tensor:
    non_nan = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    sum_values = tf.reduce_sum(non_nan, axis=axis, keepdims=keepdims)

    count = tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x))
    count_values = tf.reduce_sum(count, axis=axis, keepdims=keepdims)

    return sum_values / count_values


def tf_nan_std(
    x: tf.Tensor,
    center: tf.Tensor = None,
    axis: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> tf.Tensor:
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)

    squared_diff = tf.square(x - center)
    variance = tf_nan_mean(squared_diff, axis=axis, keepdims=keepdims)

    return tf.math.sqrt(variance)
