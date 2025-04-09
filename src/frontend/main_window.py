from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QSize
from PySide6.QtGui import QPixmap, QPainter, QCursor
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QSizePolicy,
    QButtonGroup,
    QRadioButton,
)
from ai_edge_litert.interpreter import Interpreter

from src.backend.config import (
    LOG_LEVEL,
    MODEL_SVM,
    PREDICTION_THRESHOLD_SVM,
    STABILIZATION_FRAMES,
    TRAINED_MODELS_DIR,
    MODEL_CNN_TRANSFORMER,
    INVERTED_NEW_MAPPING,
    PREDICTION_THRESHOLD_CNN_TRANSFORMER,
    SEQUENCE_LENGTH,
    SELECTED_WORDS,
    LETTERS,
    MediaPipeConfig,
    PROJECT_ROOT,
)
from src.backend.models.svm import SVM
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.camera_handler import CameraHandler
from src.backend.utils.font_utils import load_font
from src.backend.utils.mediapipe import (
    get_left_hand_landmarks,
    extract_normalized_landmarks,
    extract_structured_landmarks,
)
from src.backend.utils.training_setup import configure_gpu_memory_growth

logger = AppLogger(name=__name__, level=LOG_LEVEL)

configure_gpu_memory_growth()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rozpoznawanie języka migowego w czasie rzeczywistym")
        self.setFocusPolicy(Qt.StrongFocus)
        self.adjustSize()
        self.customFont = load_font("assets/fonts/JetBrainsMonoNL-Light.ttf")
        self.setFont(self.customFont)

        # Threshold settings for models
        self.svm_threshold = PREDICTION_THRESHOLD_SVM
        self.svm_stabilization_frames = STABILIZATION_FRAMES
        self.cnn_threshold = PREDICTION_THRESHOLD_CNN_TRANSFORMER
        self.cnn_sequence_length = SEQUENCE_LENGTH

        # MediaPipe settings for models
        self.svm_model_complexity = MediaPipeConfig.model_complexity
        self.svm_detection_confidence = MediaPipeConfig.min_detection_confidence
        self.svm_tracking_confidence = MediaPipeConfig.min_tracking_confidence

        self.cnn_model_complexity = MediaPipeConfig.model_complexity
        self.cnn_detection_confidence = MediaPipeConfig.min_detection_confidence
        self.cnn_tracking_confidence = MediaPipeConfig.min_tracking_confidence

        # Current MediaPipe settings (default for SVM)
        self.current_model_complexity = self.svm_model_complexity
        self.current_detection_confidence = self.svm_detection_confidence
        self.current_tracking_confidence = self.svm_tracking_confidence

        # Attributes for prediction (default for SVM)
        self.last_prediction = None
        self.prediction_count = 0
        self.stable_prediction = None
        self.prediction_threshold = self.svm_threshold
        self.stabilization_frames = self.svm_stabilization_frames
        self.sequence_length = self.cnn_sequence_length
        self.accumulated_text = ""

        self.model = None

        # Configuration of the main widget and horizontal layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Left: Camera view and status panel below
        camera_layout = QVBoxLayout()
        camera_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Camera view
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        self.camera_tab_widget = QTabWidget()
        self.camera_tab_widget.tabBar().hide()
        self.camera_tab_widget.setStyleSheet(
            """
            QTabBar::tab {
                height: 0;
            }
            """
        )
        self.camera_tab_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        camera_tab = QWidget()
        camera_tab_layout = QHBoxLayout()
        camera_tab_layout.addWidget(self.video_label)
        camera_tab.setLayout(camera_tab_layout)

        self.camera_tab_widget.addTab(camera_tab, "Kamera")

        camera_layout.addWidget(self.camera_tab_widget, alignment=Qt.AlignCenter)

        status_tab_widget = QTabWidget()
        status_tab_widget.tabBar().hide()
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        # QGroupBox with camera information (FPS)
        camera_info_group = QGroupBox("Informacje")
        camera_info_layout = QVBoxLayout(camera_info_group)
        fps_layout = QHBoxLayout()
        self.fps_declared_label = QLabel("Deklarowane FPS kamery: 0")
        self.fps_processing_label = QLabel("Przetwarzane FPS: 0")
        fps_layout.addWidget(self.fps_declared_label, alignment=Qt.AlignHCenter)
        fps_layout.addWidget(self.fps_processing_label, alignment=Qt.AlignHCenter)
        camera_info_layout.addLayout(fps_layout)
        camera_info_group.setLayout(camera_info_layout)
        status_layout.addWidget(camera_info_group)

        # QGroupBox with progress in prediction stabilisation
        stabilization_group = QGroupBox("Postęp stabilizacji predykcji / zbioru sekwencji")
        stabilization_layout = QVBoxLayout(stabilization_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.stabilization_frames)
        self.progress_bar.setValue(0)
        stabilization_layout.addWidget(self.progress_bar)
        stabilization_group.setLayout(stabilization_layout)
        status_layout.addWidget(stabilization_group)

        status_tab.setLayout(status_layout)
        status_tab_widget.addTab(status_tab, "Status")
        camera_layout.addWidget(status_tab_widget, alignment=Qt.AlignTop)

        # Current prediction view (read-only)
        self.predictionTextEdit = QPlainTextEdit("")
        self.predictionTextEdit.setReadOnly(True)
        self.predictionTextEdit.setPlaceholderText("Predykcja")
        self.predictionTextEdit.viewport().setCursor(QCursor(Qt.ArrowCursor))
        self.predictionTextEdit.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0); color: #ebdbb2;"
        )
        self.predictionTabWidget = QTabWidget()
        self.predictionTabWidget.setMinimumHeight(100)
        self.predictionTabWidget.addTab(self.predictionTextEdit, "")
        self.predictionTabWidget.tabBar().hide()

        # QPlainText displaying cumulative text (history)
        self.accumulatedTextEdit = QPlainTextEdit("")
        self.accumulatedTextEdit.setPlaceholderText("Wyjście")
        self.accumulatedTextEdit.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0); color: #ebdbb2;"
        )
        self.accumulatedTabWidget = QTabWidget()
        self.accumulatedTabWidget.setMinimumHeight(100)
        self.accumulatedTabWidget.addTab(self.accumulatedTextEdit, "")
        self.accumulatedTabWidget.tabBar().hide()

        # Horizontal layout for both prediction widgets
        prediction_widgets_layout = QHBoxLayout()
        prediction_widgets_layout.addWidget(self.predictionTabWidget)
        prediction_widgets_layout.addWidget(self.accumulatedTabWidget)
        camera_layout.addLayout(prediction_widgets_layout)

        # Button to clear accumulated text
        self.clear_button = QPushButton("Wyczyść")
        self.clear_button.clicked.connect(self.clear_accumulated_text)
        camera_layout.addWidget(self.clear_button)
        main_layout.addLayout(camera_layout)

        # Right part: settings
        settings_tab_widget = QTabWidget()
        settings_tab_widget.tabBar().hide()
        settings_tab = QWidget()
        settings_tab.setMinimumWidth(350)
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setAlignment(Qt.AlignTop)

        # Section for model settings
        models_group = QGroupBox("Modele")
        models_layout = QFormLayout()
        models_group.setLayout(models_layout)
        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItems(["SVM"])
        self.model_type_combobox.addItems(["CNN-Transformer"])
        self.model_type_combobox.setCurrentIndex(0)
        self.model_type_combobox.currentIndexChanged.connect(self.change_model)
        models_layout.addRow("Typ modelu:", self.model_type_combobox)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(self.prediction_threshold)
        self.threshold_spinbox.valueChanged.connect(self.update_prediction_threshold)
        self.threshold_spinbox.editingFinished.connect(self.threshold_spinbox.clearFocus)
        models_layout.addRow("Próg predykcji:", self.threshold_spinbox)

        self.stabilization_frames_spinbox = QSpinBox()
        self.stabilization_frames_spinbox.setRange(1, 120)
        self.stabilization_frames_spinbox.setSingleStep(1)
        self.stabilization_frames_spinbox.setValue(self.stabilization_frames)
        self.stabilization_frames_spinbox.valueChanged.connect(self.update_stabilization_frames)
        self.stabilization_frames_spinbox.editingFinished.connect(
            self.stabilization_frames_spinbox.clearFocus
        )
        models_layout.addRow("Ramki stabilizacyjne:", self.stabilization_frames_spinbox)

        self.sequence_length_spinbox = QSpinBox()
        self.sequence_length_spinbox.setRange(1, 120)
        self.sequence_length_spinbox.setSingleStep(1)
        self.sequence_length_spinbox.setValue(self.sequence_length)
        self.sequence_length_spinbox.valueChanged.connect(self.update_sequence_length)
        self.sequence_length_spinbox.editingFinished.connect(
            self.sequence_length_spinbox.clearFocus
        )
        models_layout.addRow("Długość sekwencji:", self.sequence_length_spinbox)

        # Button to reset model settings
        self.reset_models_button = QPushButton("Resetuj ustawienia")
        self.reset_models_button.clicked.connect(self.reset_models_settings)
        models_layout.addRow(self.reset_models_button)

        self.labels_tab_widget = QTabWidget()
        letters_widget = QWidget()
        letters_layout = QVBoxLayout(letters_widget)
        letters_edit = QPlainTextEdit()
        letters_edit.setReadOnly(True)
        letters_edit.viewport().setCursor(QCursor(Qt.ArrowCursor))
        letters_edit.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        letters_text = "\n".join([value for key, value in sorted(LETTERS.items())])
        letters_edit.setPlainText(letters_text)
        letters_layout.addWidget(letters_edit)
        self.labels_tab_widget.addTab(letters_widget, "Etykiety SVM")

        mapping_widget = QWidget()
        mapping_layout = QVBoxLayout(mapping_widget)
        mapping_edit = QPlainTextEdit()
        mapping_edit.setReadOnly(True)
        mapping_edit.viewport().setCursor(QCursor(Qt.ArrowCursor))
        mapping_edit.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        mapping_text = "\n".join(sorted(SELECTED_WORDS.keys()))
        mapping_edit.setPlainText(mapping_text)
        mapping_layout.addWidget(mapping_edit)
        self.labels_tab_widget.addTab(mapping_widget, "Etykiety CNN-Transformer")
        models_layout.addRow(self.labels_tab_widget)

        settings_layout.addWidget(models_group)

        # Section for camera settings
        general_group = QGroupBox("Kamera")
        general_layout = QHBoxLayout()
        self.camera_selector = QComboBox()
        self.toggle_button = QPushButton("Wyłącz kamerę")
        self.toggle_button.clicked.connect(self.toggle_camera)
        self.available_cameras = QMediaDevices.videoInputs()

        if not self.available_cameras:
            self.display_inactive_camera()
            self.toggle_button.setEnabled(False)
            logger.error("No cameras detected.")
        else:
            for idx, cam in enumerate(self.available_cameras):
                self.camera_selector.addItem(cam.description(), idx)

            self.camera_selector.currentIndexChanged.connect(self.switch_camera)

        general_layout.addWidget(self.camera_selector)
        general_layout.addWidget(self.toggle_button)
        general_group.setLayout(general_layout)
        settings_layout.addWidget(general_group)

        # Section for MediaPipe settings
        mediapipe_group = QGroupBox("MediaPipe")
        mediapipe_layout = QFormLayout()

        complexity_label = QLabel("Złożoność modelu:")
        self.radio0 = QRadioButton("0")
        self.radio1 = QRadioButton("1")
        self.radio2 = QRadioButton("2")
        self.radio2.setEnabled(False)
        self.model_complexity_button_group = QButtonGroup(self)
        self.model_complexity_button_group.addButton(self.radio0, 0)
        self.model_complexity_button_group.addButton(self.radio1, 1)
        self.model_complexity_button_group.addButton(self.radio2, 2)

        default_complexity = MediaPipeConfig.model_complexity
        if default_complexity == 0:
            self.radio0.setChecked(True)
        elif default_complexity == 1:
            self.radio1.setChecked(True)
        else:
            self.radio2.setChecked(True)

        self.model_complexity_button_group.buttonClicked.connect(self.on_complexity_button_clicked)

        # Layout for radio buttons
        complexity_layout = QHBoxLayout()
        complexity_layout.setAlignment(Qt.AlignLeft)
        complexity_layout.addWidget(self.radio0)
        complexity_layout.addWidget(self.radio1)
        complexity_layout.addWidget(self.radio2)
        mediapipe_layout.addRow(complexity_label, complexity_layout)

        self.detection_spinbox = QDoubleSpinBox()
        self.detection_spinbox.setRange(0.0, 1.0)
        self.detection_spinbox.setSingleStep(0.01)
        self.detection_spinbox.setValue(MediaPipeConfig.min_detection_confidence)
        self.detection_spinbox.valueChanged.connect(self.detection_value_changed)
        self.detection_spinbox.editingFinished.connect(self.detection_edit_finished)
        mediapipe_layout.addRow("Minimalna pewność wykrywania:", self.detection_spinbox)

        self.tracking_spinbox = QDoubleSpinBox()
        self.tracking_spinbox.setRange(0.0, 1.0)
        self.tracking_spinbox.setSingleStep(0.01)
        self.tracking_spinbox.setValue(MediaPipeConfig.min_tracking_confidence)
        self.tracking_spinbox.valueChanged.connect(self.tracking_value_changed)
        self.tracking_spinbox.editingFinished.connect(self.tracking_edit_finished)
        mediapipe_layout.addRow("Minimalna pewność śledzenia:", self.tracking_spinbox)

        self.draw_hands_checkbox = QCheckBox("Rysuj punkty orientacyjne dłoni")
        self.draw_hands_checkbox.setChecked(True)
        self.draw_hands_checkbox.stateChanged.connect(self.toggle_drawing_hands)
        mediapipe_layout.addRow(self.draw_hands_checkbox)

        self.draw_face_checkbox = QCheckBox("Rysuj punkty orientacyjne twarzy")
        self.draw_face_checkbox.setChecked(True)
        self.draw_face_checkbox.stateChanged.connect(self.toggle_drawing_face)
        mediapipe_layout.addRow(self.draw_face_checkbox)

        self.draw_pose_checkbox = QCheckBox("Rysuj punkty orientacyjne postaci")
        self.draw_pose_checkbox.setChecked(True)
        self.draw_pose_checkbox.stateChanged.connect(self.toggle_drawing_pose)
        mediapipe_layout.addRow(self.draw_pose_checkbox)

        self.reset_button = QPushButton("Resetuj ustawienia")
        self.reset_button.clicked.connect(self.reset_mediapipe_settings)
        mediapipe_layout.addRow(self.reset_button)

        mediapipe_group.setLayout(mediapipe_layout)
        settings_layout.addWidget(mediapipe_group)

        settings_tab_widget.addTab(settings_tab, "Ustawienia")
        main_layout.addWidget(settings_tab_widget)

        # backend: camera handling
        self.camera_handler = CameraHandler()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.sequence_started = False
        self.landmark_buffer = []

        self.load_default_model()

    def load_default_model(self):
        default_model_type = self.model_type_combobox.itemText(0)
        self.initialize_model(default_model_type)

    def update_model_complexity_options(self, model_type):
        self.radio2.setEnabled(model_type == "CNN-Transformer")
        if model_type != "CNN-Transformer" and self.radio2.isChecked():
            self.radio0.setChecked(True)

    def change_model(self):
        selected_model = self.model_type_combobox.currentText()
        self.update_model_complexity_options(selected_model)
        self.initialize_model(selected_model)

    def initialize_model(self, model_type):
        project_root = Path(__file__).resolve().parent.parent.parent
        if model_type == "SVM":
            model_path = project_root / TRAINED_MODELS_DIR / MODEL_SVM
            try:
                self.model = SVM()
                self.model.load_model(model_path)
                logger.info("SVM model loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Nie udało się załadować modelu SVM:\n{e}")
                logger.error(f"Failed to load SVM model: {e}")
                exit(1)

            # Settings specific to SVM
            model_params = {
                "model_complexity": self.svm_model_complexity,
                "detection_confidence": self.svm_detection_confidence,
                "tracking_confidence": self.svm_tracking_confidence,
                "prediction_threshold": self.svm_threshold,
                "progress_range": self.svm_stabilization_frames,
            }
            ui_settings = {
                "enable_stabilization": True,
                "enable_sequence_length": False,
                "stabilization_value": self.svm_stabilization_frames,
            }
            self.sequence_length = None

        elif model_type == "CNN-Transformer":
            model_path = project_root / TRAINED_MODELS_DIR / MODEL_CNN_TRANSFORMER
            try:
                self.model = Interpreter(model_path=str(model_path))
                logger.info("CNN-Transformer (TFLite) model loaded successfully.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się załadować modelu CNN-Transformer:\n{e}"
                )
                logger.error(f"Failed to load CNN-Transformer model: {e}")
                exit(1)

            # Settings specific to CNN-Transformer
            model_params = {
                "model_complexity": self.cnn_model_complexity,
                "detection_confidence": self.cnn_detection_confidence,
                "tracking_confidence": self.cnn_tracking_confidence,
                "prediction_threshold": self.cnn_threshold,
                "progress_range": self.cnn_sequence_length,
            }
            ui_settings = {
                "enable_stabilization": False,
                "enable_sequence_length": True,
                "sequence_length_value": self.cnn_sequence_length,
            }
        else:
            logger.error(f"Unsupported model type: {model_type}")
            self.model = None
            return

        # Common settings for both models
        self.current_model_complexity = model_params["model_complexity"]
        self.current_detection_confidence = model_params["detection_confidence"]
        self.current_tracking_confidence = model_params["tracking_confidence"]

        if self.current_model_complexity == 0:
            self.radio0.setChecked(True)
        elif self.current_model_complexity == 1:
            self.radio1.setChecked(True)
        else:
            self.radio2.setChecked(True)
        self.detection_spinbox.setValue(self.current_detection_confidence)
        self.tracking_spinbox.setValue(self.current_tracking_confidence)

        self.prediction_threshold = model_params["prediction_threshold"]
        self.threshold_spinbox.setValue(model_params["prediction_threshold"])
        self.progress_bar.setRange(0, model_params["progress_range"])

        # Settings dependent on model type
        if ui_settings.get("enable_stabilization"):
            self.stabilization_frames = model_params["progress_range"]
            self.stabilization_frames_spinbox.setEnabled(True)
            self.stabilization_frames_spinbox.setValue(ui_settings["stabilization_value"])
            self.sequence_length_spinbox.setEnabled(False)
        elif ui_settings.get("enable_sequence_length"):
            self.sequence_length = ui_settings["sequence_length_value"]
            self.sequence_length_spinbox.setEnabled(True)
            self.sequence_length_spinbox.setValue(ui_settings["sequence_length_value"])
            self.stabilization_frames_spinbox.setEnabled(False)

        # Drawing settings
        if model_type == "SVM":
            self.draw_hands_checkbox.setEnabled(True)
            self.draw_face_checkbox.setEnabled(False)
            self.draw_pose_checkbox.setEnabled(False)
        elif model_type == "CNN-Transformer":
            self.draw_hands_checkbox.setEnabled(True)
            self.draw_face_checkbox.setEnabled(True)
            self.draw_pose_checkbox.setEnabled(True)

    def update_frame(self):
        if not self.camera_handler.camera_active:
            self.display_inactive_camera()
            return

        if isinstance(self.model, SVM):
            frame = self.camera_handler.get_frame()
        elif isinstance(self.model, Interpreter):
            frame = self.camera_handler.get_frame_holistic()
        else:
            frame = None

        if frame is None:
            return

        if len(frame) == 4:
            qimg, real_fps, fps, mp_results = frame
        else:
            qimg, real_fps, fps = frame
            mp_results = None

        if qimg is not None:
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
            if isinstance(self.model, SVM):
                if mp_results and mp_results.multi_hand_landmarks and mp_results.multi_handedness:
                    left_hand_landmarks = get_left_hand_landmarks(mp_results)
                    if left_hand_landmarks:
                        features = extract_normalized_landmarks(
                            left_hand_landmarks, qimg.width(), qimg.height(), padding=20
                        )

                        features = np.array(features).reshape(1, -1)

                        self.process_prediction(features)
                    else:
                        self.reset_prediction_state("Predykcja: brak lewej dłoni")
                else:
                    self.reset_prediction_state("Predykcja: brak wykrytych dłoni")
            elif isinstance(self.model, Interpreter):
                if not self.sequence_started:
                    if mp_results and (
                        mp_results.left_hand_landmarks or mp_results.right_hand_landmarks
                    ):
                        self.sequence_started = True
                if self.sequence_started:
                    if mp_results:
                        landmarks = extract_structured_landmarks(mp_results)
                        self.landmark_buffer.append(landmarks)
                        self.progress_bar.setValue(len(self.landmark_buffer))

                        if len(self.landmark_buffer) == self.sequence_length:
                            features = np.array(self.landmark_buffer, dtype=np.float32)
                            self.process_prediction(features)
                            self.landmark_buffer = []
                            self.progress_bar.setValue(0)
                            self.sequence_started = False

        if fps > 0:
            self.fps_declared_label.setText(f"Deklarowane FPS kamery: {real_fps:.2f}")
            self.fps_processing_label.setText(f"Przetwarzane FPS: {fps:.2f}")

    def display_inactive_camera(self):
        svg_path = str(PROJECT_ROOT / "assets/icons/webcam-off.svg")
        svg_renderer = QSvgRenderer(svg_path)

        label_rect = self.video_label.rect()

        pixmap = QPixmap(label_rect.size())
        pixmap.fill(Qt.transparent)

        svg_default_size = svg_renderer.defaultSize()
        if not svg_default_size.isEmpty():
            scaled_size = svg_default_size.scaled(label_rect.size(), Qt.KeepAspectRatio)
            factor = 0.5
            new_width = int(scaled_size.width() * factor)
            new_height = int(scaled_size.height() * factor)
            scaled_size = QSize(new_width, new_height)
            x = (label_rect.width() - scaled_size.width()) // 2
            y = (label_rect.height() - scaled_size.height()) // 2
            target_rect = QRect(x, y, scaled_size.width(), scaled_size.height())
        else:
            target_rect = label_rect

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        svg_renderer.render(painter, target_rect)
        painter.end()

        self.video_label.setPixmap(pixmap)
        self.fps_declared_label.setText("Deklarowane FPS kamery: 0")
        self.fps_processing_label.setText("Przetwarzane FPS: 0")

    def process_prediction(self, features):
        try:
            if isinstance(self.model, SVM):
                proba = self.model.predict_proba(features)
                max_proba = np.max(proba)
                new_prediction = self.model.predict(features)[0]

                if max_proba >= self.prediction_threshold:
                    self.update_prediction(new_prediction, max_proba)
                else:
                    self.predictionTextEdit.setPlainText(
                        f"Predykcja niepewna: {new_prediction} (pewność: {max_proba:.2f})"
                    )
                    self.prediction_count = max(self.prediction_count - 1, 0)
                    self.progress_bar.setValue(self.prediction_count)
            if isinstance(self.model, Interpreter):
                input_details = self.model.get_input_details()

                if input_details[0]["shape"][0] != features.shape[0]:
                    self.model.resize_tensor_input(input_details[0]["index"], features.shape)
                    self.model.allocate_tensors()

                self.model.set_tensor(input_details[0]["index"], features)
                self.model.invoke()
                outputs = self.model.get_tensor(self.model.get_output_details()[0]["index"])

                predicted_class = np.argmax(outputs, axis=1)[0]
                confidence = outputs[0][predicted_class]

                if confidence >= self.prediction_threshold:
                    self.update_prediction(predicted_class, float(confidence))
                else:
                    self.predictionTextEdit.setPlainText(
                        f"Predykcja niepewna: {INVERTED_NEW_MAPPING[predicted_class]} ({float(confidence):.2f})"
                    )
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            self.predictionTextEdit.setPlainText("Predykcja: błąd")

    def update_prediction(self, new_prediction, max_proba):
        if isinstance(self.model, SVM):
            if self.last_prediction is None or new_prediction != self.last_prediction:
                self.last_prediction = new_prediction
                self.prediction_count = 1
            else:
                self.prediction_count += 1

            self.progress_bar.setValue(self.prediction_count)

            self.predictionTextEdit.setPlainText(
                f"Predykcja: {new_prediction} (pewność: {max_proba:.2f})"
            )

            if self.prediction_count >= self.stabilization_frames:
                self.update_accumulated_text(new_prediction)
                self.reset_prediction_state()
        elif isinstance(self.model, Interpreter):
            self.predictionTextEdit.setPlainText(
                f"Predykcja: {INVERTED_NEW_MAPPING[new_prediction]} ({max_proba:.2f})"
            )
            self.update_accumulated_text(INVERTED_NEW_MAPPING[new_prediction])

    def update_accumulated_text(self, new_prediction):
        if isinstance(self.model, SVM):
            if new_prediction == "BACKSPACE":
                if self.accumulated_text:
                    self.accumulated_text = self.accumulated_text[:-1]
            elif new_prediction == "SPACE":
                self.accumulated_text += " "
            else:
                self.accumulated_text += new_prediction
            self.accumulatedTextEdit.setPlainText(self.accumulated_text)
        elif isinstance(self.model, Interpreter):
            self.accumulated_text += new_prediction + " "
            self.accumulatedTextEdit.setPlainText(self.accumulated_text)

    def reset_prediction_state(self, message=None):
        if message:
            self.predictionTextEdit.setPlainText(message)
        self.last_prediction = None
        self.prediction_count = 0
        self.progress_bar.setValue(0)

    def switch_camera(self, index):
        self.camera_handler.switch_camera(index)
        self.toggle_button.setText("Wyłącz kamerę")

    def toggle_camera(self):
        self.camera_handler.toggle_camera()
        if self.camera_handler.camera_active:
            self.toggle_button.setText("Wyłącz kamerę")
        else:
            self.toggle_button.setText("Włącz kamerę")

    def toggle_drawing_hands(self):
        self.camera_handler.set_draw_hands_landmarks(self.draw_hands_checkbox.isChecked())

    def toggle_drawing_face(self):
        self.camera_handler.set_draw_face_landmarks(self.draw_face_checkbox.isChecked())

    def toggle_drawing_pose(self):
        self.camera_handler.set_draw_pose_landmarks(self.draw_pose_checkbox.isChecked())

    def update_mediapipe_settings(self):
        try:
            new_complexity = self.model_complexity_button_group.checkedId()
            new_detection = self.detection_spinbox.value()
            new_tracking = self.tracking_spinbox.value()
            self.current_model_complexity = new_complexity
            self.current_detection_confidence = new_detection
            self.current_tracking_confidence = new_tracking
            self.camera_handler.set_mediapipe_settings(
                new_complexity, new_detection, new_tracking, self.model
            )

            if self.model_type_combobox.currentText() == "SVM":
                self.svm_model_complexity = new_complexity
                self.svm_detection_confidence = new_detection
                self.svm_tracking_confidence = new_tracking
            elif self.model_type_combobox.currentText() == "CNN-Transformer":
                self.cnn_model_complexity = new_complexity
                self.cnn_detection_confidence = new_detection
                self.cnn_tracking_confidence = new_tracking

            logger.info("MediaPipe settings updated.")
        except Exception as e:
            logger.error(f"Failed to update MediaPipe settings: {e}")

    def reset_models_settings(self):
        try:
            self.model_type_combobox.blockSignals(True)
            self.threshold_spinbox.blockSignals(True)
            self.stabilization_frames_spinbox.blockSignals(True)
            self.sequence_length_spinbox.blockSignals(True)
            if self.model_type_combobox.currentText() == "SVM":
                self.svm_threshold = PREDICTION_THRESHOLD_SVM
                self.svm_stabilization_frames = STABILIZATION_FRAMES
                self.threshold_spinbox.setValue(PREDICTION_THRESHOLD_SVM)
                self.stabilization_frames_spinbox.setValue(STABILIZATION_FRAMES)
            elif self.model_type_combobox.currentText() == "CNN-Transformer":
                self.cnn_threshold = PREDICTION_THRESHOLD_CNN_TRANSFORMER
                self.cnn_sequence_length = SEQUENCE_LENGTH
                self.threshold_spinbox.setValue(PREDICTION_THRESHOLD_CNN_TRANSFORMER)
                self.sequence_length_spinbox.setValue(SEQUENCE_LENGTH)
            logger.info("Model settings have been reset.")
        except Exception as e:
            logger.error(f"Failed to reset model settings: {e}")
        finally:
            self.model_type_combobox.blockSignals(False)
            self.threshold_spinbox.blockSignals(False)
            self.stabilization_frames_spinbox.blockSignals(False)
            self.sequence_length_spinbox.blockSignals(False)

    def reset_mediapipe_settings(self):
        try:
            self.model_complexity_button_group.blockSignals(True)
            self.detection_spinbox.blockSignals(True)
            self.tracking_spinbox.blockSignals(True)

            if self.model_type_combobox.currentText() == "SVM":
                self.svm_model_complexity = MediaPipeConfig.model_complexity
                self.svm_detection_confidence = MediaPipeConfig.min_detection_confidence
                self.svm_tracking_confidence = MediaPipeConfig.min_tracking_confidence

                if self.svm_model_complexity == 0:
                    self.radio0.setChecked(True)
                elif self.svm_model_complexity == 1:
                    self.radio1.setChecked(True)
                else:
                    self.radio2.setChecked(True)
                self.detection_spinbox.setValue(self.svm_detection_confidence)
                self.tracking_spinbox.setValue(self.svm_tracking_confidence)
            elif self.model_type_combobox.currentText() == "CNN-Transformer":
                self.cnn_model_complexity = MediaPipeConfig.model_complexity
                self.cnn_detection_confidence = MediaPipeConfig.min_detection_confidence
                self.cnn_tracking_confidence = MediaPipeConfig.min_tracking_confidence

                if self.cnn_model_complexity == 0:
                    self.radio0.setChecked(True)
                elif self.cnn_model_complexity == 1:
                    self.radio1.setChecked(True)
                else:
                    self.radio2.setChecked(True)
                self.detection_spinbox.setValue(self.cnn_detection_confidence)
                self.tracking_spinbox.setValue(self.cnn_tracking_confidence)

            self.draw_hands_checkbox.setChecked(True)
            self.draw_face_checkbox.setChecked(True)
            self.draw_pose_checkbox.setChecked(True)

            logger.info("MediaPipe settings have been reset.")
        except Exception as e:
            logger.error(f"Failed to reset MediaPipe settings: {e}")
        finally:
            self.model_complexity_button_group.blockSignals(False)
            self.detection_spinbox.blockSignals(False)
            self.tracking_spinbox.blockSignals(False)
        self.update_mediapipe_settings()

    def update_prediction_threshold(self, new_value):
        self.prediction_threshold = new_value
        if self.model_type_combobox.currentText() == "SVM":
            self.svm_threshold = new_value
        elif self.model_type_combobox.currentText() == "CNN-Transformer":
            self.cnn_threshold = new_value

    def update_stabilization_frames(self, new_value):
        self.stabilization_frames = new_value
        if self.model_type_combobox.currentText() == "SVM":
            self.svm_stabilization_frames = new_value
        self.progress_bar.setRange(0, new_value)

    def update_sequence_length(self, new_value):
        self.sequence_length = new_value
        if self.model_type_combobox.currentText() == "CNN-Transformer":
            self.cnn_sequence_length = new_value
        self.progress_bar.setRange(0, new_value)

    def closeEvent(self, event):
        self.camera_handler.close()
        event.accept()

    def on_complexity_button_clicked(self):
        self.update_mediapipe_settings()

    def detection_value_changed(self):
        self.update_mediapipe_settings()

    def tracking_value_changed(self):
        self.update_mediapipe_settings()

    def detection_edit_finished(self):
        self.detection_spinbox.clearFocus()

    def tracking_edit_finished(self):
        self.tracking_spinbox.clearFocus()

    def clear_accumulated_text(self):
        self.accumulated_text = ""
        self.accumulatedTextEdit.clear()
