import time

import cv2
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap

import mediapipe as mp
from src.backend.config import CAMERA_CONFIG, DRAWING_CONFIG, MEDIAPIPE_SETTINGS


def draw_custom_landmarks(qimg, hand_landmarks, connections, w, h):
    pixmap = QPixmap.fromImage(qimg)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    try:
        landmark_color = DRAWING_CONFIG["landmark_color"]
        r, g, b = landmark_color

        point_pen = QPen(QColor(r, g, b))
        point_pen.setWidth(DRAWING_CONFIG["landmark_thickness"])
        point_brush = QBrush(QColor(r, g, b))
        radius = DRAWING_CONFIG["landmark_radius"]

        connection_pen = QPen(QColor(r, g, b))
        connection_pen.setWidth(DRAWING_CONFIG["connection_thickness"])

        painter.setPen(connection_pen)
        for connection in connections:
            start_idx, end_idx = connection
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            x1 = int(start_point.x * w)
            y1 = int(start_point.y * h)
            x2 = int(end_point.x * w)
            y2 = int(end_point.y * h)
            painter.drawLine(x1, y1, x2, y2)

        painter.setPen(point_pen)
        painter.setBrush(point_brush)
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
    finally:
        painter.end()

    return pixmap.toImage()


class CameraHandler:
    def __init__(
        self,
        camera_index=CAMERA_CONFIG["index"],
        model_complexity=MEDIAPIPE_SETTINGS["model_complexity"],
        min_detection_confidence=MEDIAPIPE_SETTINGS["min_detection_confidence"],
        min_tracking_confidence=MEDIAPIPE_SETTINGS["min_tracking_confidence"],
    ):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc(*CAMERA_CONFIG["video_codec"]),
        )
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        self.camera_active = CAMERA_CONFIG["camera_active"]

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
        self.mp_draw = mp.solutions.drawing_utils

        self.draw_landmarks = True

        self.frame_count = 0
        self.last_fps_update_time = time.perf_counter()
        self.fps = 0.0

    def get_frame(self):
        if self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return None, 0.0, 0.0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.medianBlur(frame_rgb, 1)
            frame_rgb = cv2.flip(frame_rgb, 1)

            frame_rgb.flags.writeable = False
            results = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            if results.multi_hand_landmarks and self.draw_landmarks:
                for handLms in results.multi_hand_landmarks:
                    qimg = draw_custom_landmarks(
                        qimg, handLms, self.mp_hands.HAND_CONNECTIONS, w, h
                    )

            declared_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - self.last_fps_update_time

            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_update_time = current_time

            return qimg, declared_fps, self.fps, results

        return None, 0.0, 0.0

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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.camera_active = True

    def set_draw_landmarks(self, flag: bool):
        self.draw_landmarks = flag

    def set_mediapipe_settings(
        self,
        model_complexity,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self):
        if self.cap:
            self.cap.release()
