import time

import cv2
import tensorflow as tf
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap

import mediapipe as mp
from src.backend.config import DrawingConfig, MediaPipeConfig, CameraConfig
from src.backend.models.svm import SVM


def setup_painter(qimg):
    pixmap = QPixmap.fromImage(qimg)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    return pixmap, painter


def setup_pen_and_brush(config):
    landmark_pen = QPen(config["landmark_color"])
    landmark_pen.setWidth(config["landmark_thickness"])
    landmark_brush = QBrush(config["landmark_color"])

    connection_pen = QPen(config["connection_color"])
    connection_pen.setWidth(config["connection_thickness"])

    return landmark_pen, landmark_brush, connection_pen


def get_drawing_config():
    return {
        "landmark_color": QColor(
            DrawingConfig.landmark_color[0],
            DrawingConfig.landmark_color[1],
            DrawingConfig.landmark_color[2],
        ),
        "landmark_thickness": DrawingConfig.landmark_thickness,
        "landmark_radius": DrawingConfig.landmark_radius,
        "connection_color": QColor(
            DrawingConfig.connection_color[0],
            DrawingConfig.connection_color[1],
            DrawingConfig.connection_color[2],
        ),
        "connection_thickness": DrawingConfig.connection_thickness,
    }


def draw_landmark(painter, lm, w, h, radius):
    x = int(lm.x * w)
    y = int(lm.y * h)
    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)


def draw_connections(painter, landmarks, connections, w, h):
    for start_idx, end_idx in connections:
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        x1, y1 = int(start_point.x * w), int(start_point.y * h)
        x2, y2 = int(end_point.x * w), int(end_point.y * h)
        painter.drawLine(x1, y1, x2, y2)


def draw_custom_landmarks(qimg, hand_landmarks, connections, w, h):
    pixmap, painter = setup_painter(qimg)
    config = get_drawing_config()
    landmark_pen, landmark_brush, connection_pen = setup_pen_and_brush(config)

    try:
        painter.setPen(connection_pen)
        draw_connections(painter, hand_landmarks, connections, w, h)

        painter.setPen(landmark_pen)
        painter.setBrush(landmark_brush)
        for lm in hand_landmarks.landmark:
            draw_landmark(painter, lm, w, h, config["landmark_radius"])
    finally:
        painter.end()

    return pixmap.toImage()


def draw_all_landmarks(qimg, results, w, h, draw_hand=True, draw_face=True, draw_pose=True):
    pixmap, painter = setup_painter(qimg)
    config = get_drawing_config()
    landmark_pen, landmark_brush, connection_pen = setup_pen_and_brush(config)

    if draw_hand:
        if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks:
            painter.setPen(connection_pen)
            draw_connections(
                painter, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, w, h
            )
            painter.setPen(landmark_pen)
            painter.setBrush(landmark_brush)
            for lm in results.left_hand_landmarks.landmark:
                draw_landmark(painter, lm, w, h, config["landmark_radius"])

        if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks:
            painter.setPen(connection_pen)
            draw_connections(
                painter, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, w, h
            )
            painter.setPen(landmark_pen)
            painter.setBrush(landmark_brush)
            for lm in results.right_hand_landmarks.landmark:
                draw_landmark(painter, lm, w, h, config["landmark_radius"])

    if draw_face and hasattr(results, "face_landmarks") and results.face_landmarks:
        painter.setPen(landmark_pen)
        painter.setBrush(landmark_brush)
        for lm in results.face_landmarks.landmark:
            draw_landmark(painter, lm, w, h, config["landmark_radius"] - 1)

    if draw_pose and hasattr(results, "pose_landmarks") and results.pose_landmarks:
        painter.setPen(connection_pen)
        draw_connections(painter, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, w, h)
        painter.setPen(landmark_pen)
        painter.setBrush(landmark_brush)
        for lm in results.pose_landmarks.landmark:
            draw_landmark(painter, lm, w, h, config["landmark_radius"])

    painter.end()
    return pixmap.toImage()


class CameraHandler:
    def __init__(
        self,
        camera_index=CameraConfig.index,
        model_complexity=MediaPipeConfig.model_complexity,
        min_detection_confidence=MediaPipeConfig.min_detection_confidence,
        min_tracking_confidence=MediaPipeConfig.min_tracking_confidence,
    ):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.height)
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc(*CameraConfig.video_codec),
        )
        self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.fps)
        self.camera_active = CameraConfig.camera_active

        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            smooth_landmarks=True,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.mp_draw = mp.solutions.drawing_utils

        self.draw_hands_landmarks = True
        self.draw_face_landmarks = True
        self.draw_pose_landmarks = True

        self.frame_count = 0
        self.last_fps_update_time = time.perf_counter()
        self.fps = 0.0

    def get_frame(self):
        qimg, results, w, h = self._capture_and_process(self.hands.process)

        if qimg is None:
            return None, 0.0, 0.0, results

        if results.multi_hand_landmarks and self.draw_hands_landmarks:
            for handLms in results.multi_hand_landmarks:
                qimg = draw_custom_landmarks(qimg, handLms, self.mp_hands.HAND_CONNECTIONS, w, h)

        declared_fps, self.fps = self._update_fps()

        return qimg, declared_fps, self.fps, results

    def get_frame_holistic(self):
        qimg, results, w, h = self._capture_and_process(self.holistic.process)

        if qimg is None:
            return None, 0.0, 0.0, results

        if self.draw_hands_landmarks or self.draw_face_landmarks or self.draw_pose_landmarks:
            qimg = draw_all_landmarks(
                qimg,
                results,
                w,
                h,
                self.draw_hands_landmarks,
                self.draw_face_landmarks,
                self.draw_pose_landmarks,
            )

        declared_fps, self.fps = self._update_fps()

        return qimg, declared_fps, self.fps, results

    def _capture_and_process(self, process_fn):
        if not (self.camera_active and self.cap.isOpened()):
            return None, None, None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)

        frame_rgb.flags.writeable = False
        results = process_fn(frame_rgb)
        frame_rgb.flags.writeable = True

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        return qimg, results, w, h

    def _update_fps(self):
        declared_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed = current_time - self.last_fps_update_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update_time = current_time

        return declared_fps, self.fps

    def switch_camera(self, new_index):
        if self.cap:
            self.cap.release()
        self.camera_index = new_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.camera_active = True

    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
        else:
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.height)
                self.cap.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*CameraConfig.video_codec)
                )
                self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.fps)
            self.camera_active = True

    def set_draw_hands_landmarks(self, flag: bool):
        self.draw_hands_landmarks = flag

    def set_draw_face_landmarks(self, flag: bool):
        self.draw_face_landmarks = flag

    def set_draw_pose_landmarks(self, flag: bool):
        self.draw_pose_landmarks = flag

    def set_mediapipe_settings(
        self,
        model_complexity,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if isinstance(model, SVM):
            self.hands.close()
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

        if isinstance(model, tf.lite.Interpreter):
            self.holistic.close()
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                smooth_landmarks=True,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def close(self):
        if self.cap:
            self.cap.release()

        if self.hands:
            self.hands.close()

        if self.holistic:
            self.holistic.close()
