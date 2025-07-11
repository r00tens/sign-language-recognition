import numpy as np


def get_left_hand_landmarks(mp_results):
    for landmarks, handedness in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
        if handedness.classification[0].label == "Left":
            return landmarks
    return None


def extract_normalized_landmarks(hand_landmarks, image_width, image_height, padding=20):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(xs) * image_width), int(max(xs) * image_width)
    y_min, y_max = int(min(ys) * image_height), int(max(ys) * image_height)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    landmarks = []
    for lm in hand_landmarks.landmark:
        pixel_x = lm.x * image_width
        pixel_y = lm.y * image_height

        rel_x = (pixel_x - x_min) / bbox_width if bbox_width > 0 else 0
        rel_y = (pixel_y - y_min) / bbox_height if bbox_height > 0 else 0

        landmarks.extend([rel_x, rel_y, lm.z])

    return landmarks


def extract_structured_landmarks(mp_results):
    if mp_results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in mp_results.face_landmarks.landmark], dtype=np.float32
        )
    else:
        face = np.full(shape=(468, 3), fill_value=np.nan, dtype=np.float32)

    if mp_results.left_hand_landmarks:
        left = np.array(
            [[lm.x, lm.y, lm.z] for lm in mp_results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        left = np.full(shape=(21, 3), fill_value=np.nan, dtype=np.float32)

    if mp_results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in mp_results.pose_landmarks.landmark], dtype=np.float32
        )
    else:
        pose = np.full(shape=(33, 3), fill_value=np.nan, dtype=np.float32)

    if mp_results.right_hand_landmarks:
        right = np.array(
            [[lm.x, lm.y, lm.z] for lm in mp_results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        right = np.full(shape=(21, 3), fill_value=np.nan, dtype=np.float32)

    all_landmarks = np.concatenate([face, left, pose, right], axis=0)
    return all_landmarks
