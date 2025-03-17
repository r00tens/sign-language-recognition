from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QMediaDevices
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
)

from src.backend.config import (
    LOG_LEVEL,
    MEDIAPIPE_SETTINGS,
    MODEL_SVM,
    PREDICTION_THRESHOLD,
    STABILIZATION_FRAMES,
    TRAINED_MODELS_DIR,
)
from src.backend.models.svm import SVM
from src.backend.utils.app_logger import AppLogger
from src.backend.utils.camera_handler import CameraHandler
from src.backend.utils.font_utilities import load_font
from src.backend.utils.mediapipe import get_left_hand_landmarks, extract_normalized_landmarks

logger = AppLogger(name=__name__, level=LOG_LEVEL)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rozpoznawanie języka migowego")
        self.setFocusPolicy(Qt.StrongFocus)
        self.adjustSize()
        self.customFont = load_font("assets/fonts/JetBrainsMonoNL-Light.ttf")
        self.setFont(self.customFont)

        # Atrybuty stabilizacji predykcji
        self.last_prediction = None
        self.prediction_count = 0
        self.stable_prediction = None
        self.prediction_threshold = PREDICTION_THRESHOLD
        self.stabilization_frames = STABILIZATION_FRAMES

        # Atrybuty do budowania tekstu
        self.accumulated_text = ""

        self.model = None

        # Konfiguracja głównego widgetu i układu poziomego
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Lewa część: widok z kamery i panel statusu poniżej
        camera_layout = QVBoxLayout()
        camera_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Widok kamery
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        camera_layout.addWidget(self.video_label, alignment=Qt.AlignHCenter)

        # QTabWidget z pojedynczą zakładką "Status", zawierającą oba GroupBoxy
        status_tab_widget = QTabWidget()
        status_tab_widget.tabBar().hide()
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        # GroupBox z informacjami o kamerze (FPS)
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

        # GroupBox z postępem stabilizacji predykcji
        stabilization_group = QGroupBox("Postęp stabilizacji predykcji")
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

        # Widok bieżącej predykcji (tylko do odczytu)
        self.predictionTextEdit = QPlainTextEdit("")
        self.predictionTextEdit.setReadOnly(True)
        self.predictionTextEdit.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.predictionTabWidget = QTabWidget()
        self.predictionTabWidget.addTab(self.predictionTextEdit, "")
        self.predictionTabWidget.tabBar().hide()

        # Pole tekstowe wyświetlające zbudowany tekst (historia)
        self.accumulatedTextEdit = QPlainTextEdit("")
        self.accumulatedTextEdit.setPlaceholderText("Wyjście")
        self.accumulatedTextEdit.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0); color: #ebdbb2;"
        )
        self.accumulatedTabWidget = QTabWidget()
        self.accumulatedTabWidget.addTab(self.accumulatedTextEdit, "")
        self.accumulatedTabWidget.tabBar().hide()

        # Układ poziomy dla obu widgetów predykcji
        prediction_widgets_layout = QHBoxLayout()
        prediction_widgets_layout.addWidget(self.predictionTabWidget)
        prediction_widgets_layout.addWidget(self.accumulatedTabWidget)

        # Dodanie układu predykcji do głównego układu kamery
        camera_layout.addLayout(prediction_widgets_layout)

        # Przycisk do wyczyszczenia zbudowanego tekstu
        self.clear_button = QPushButton("Wyczyść")
        self.clear_button.clicked.connect(self.clear_accumulated_text)
        camera_layout.addWidget(self.clear_button)

        main_layout.addLayout(camera_layout)

        # Prawa część: zakładka z ustawieniami
        settings_tab_widget = QTabWidget()
        settings_tab_widget.tabBar().hide()
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setAlignment(Qt.AlignTop)

        # Sekcja ustawień modeli
        models_group = QGroupBox("Modele")
        models_layout = QFormLayout()
        models_group.setLayout(models_layout)
        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItems(["SVM"])
        self.model_type_combobox.setCurrentIndex(0)
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

        # Przycisk do resetowania ustawień sekcji "Modele"
        self.reset_models_button = QPushButton("Resetuj ustawienia")
        self.reset_models_button.clicked.connect(self.reset_models_settings)
        models_layout.addRow(self.reset_models_button)

        settings_layout.addWidget(models_group)

        # Sekcja ustawień kamery
        general_group = QGroupBox("Kamera")
        general_layout = QHBoxLayout()
        self.camera_selector = QComboBox()
        self.toggle_button = QPushButton("Wyłącz kamerę")
        self.toggle_button.clicked.connect(self.toggle_camera)
        self.available_cameras = QMediaDevices.videoInputs()
        if not self.available_cameras:
            black_pixmap = QPixmap(self.video_label.width(), self.video_label.height())
            black_pixmap.fill(Qt.black)
            self.video_label.setPixmap(black_pixmap)
            self.toggle_button.setEnabled(False)
        else:
            for idx, cam in enumerate(self.available_cameras):
                self.camera_selector.addItem(cam.description(), idx)
            self.camera_selector.currentIndexChanged.connect(self.switch_camera)

        general_layout.addWidget(self.camera_selector)
        general_layout.addWidget(self.toggle_button)
        general_group.setLayout(general_layout)
        settings_layout.addWidget(general_group)

        # Sekcja ustawień MediaPipe
        mediapipe_group = QGroupBox("MediaPipe")
        mediapipe_layout = QFormLayout()

        self.model_complexity_combobox = QComboBox()
        self.model_complexity_combobox.addItems(["0", "1"])
        self.model_complexity_combobox.setCurrentText(str(MEDIAPIPE_SETTINGS["model_complexity"]))
        self.model_complexity_combobox.currentIndexChanged.connect(
            self.model_complexity_current_index_changed
        )
        mediapipe_layout.addRow("Złożoność modelu:", self.model_complexity_combobox)

        self.detection_spinbox = QDoubleSpinBox()
        self.detection_spinbox.setRange(0.0, 1.0)
        self.detection_spinbox.setSingleStep(0.01)
        self.detection_spinbox.setValue(MEDIAPIPE_SETTINGS["min_detection_confidence"])
        self.detection_spinbox.valueChanged.connect(self.detection_value_changed)
        self.detection_spinbox.editingFinished.connect(self.detection_edit_finished)
        mediapipe_layout.addRow("Minimalna pewność wykrywania:", self.detection_spinbox)

        self.tracking_spinbox = QDoubleSpinBox()
        self.tracking_spinbox.setRange(0.0, 1.0)
        self.tracking_spinbox.setSingleStep(0.01)
        self.tracking_spinbox.setValue(MEDIAPIPE_SETTINGS["min_tracking_confidence"])
        self.tracking_spinbox.valueChanged.connect(self.tracking_value_changed)
        self.tracking_spinbox.editingFinished.connect(self.tracking_edit_finished)
        mediapipe_layout.addRow("Minimalna pewność śledzenia:", self.tracking_spinbox)

        self.draw_checkbox = QCheckBox("Rysuj punkty orientacyjne dłoni")
        self.draw_checkbox.setChecked(True)
        self.draw_checkbox.stateChanged.connect(self.toggle_drawing)
        mediapipe_layout.addRow(self.draw_checkbox)

        self.reset_button = QPushButton("Resetuj ustawienia")
        self.reset_button.clicked.connect(self.reset_mediapipe_settings)
        mediapipe_layout.addRow(self.reset_button)

        mediapipe_group.setLayout(mediapipe_layout)
        settings_layout.addWidget(mediapipe_group)

        settings_tab_widget.addTab(settings_tab, "Ustawienia")
        main_layout.addWidget(settings_tab_widget)

        # Backend: obsługa kamery
        self.camera_handler = CameraHandler()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.load_default_model()

    def load_default_model(self):
        if self.model_type_combobox.count() == 0:
            logger.error("Brak opcji modelu w comboboxie!")
            return

        default_model_type = self.model_type_combobox.itemText(0)
        self.initialize_model(default_model_type)

    def change_model(self):
        selected_model = self.model_type_combobox.currentText()
        self.initialize_model(selected_model)

    def initialize_model(self, model_type):
        project_root = Path(__file__).resolve().parent.parent.parent

        if model_type == "SVM":
            model_path = project_root / TRAINED_MODELS_DIR / MODEL_SVM
            self.model = SVM()

            try:
                self.model.load_model(model_path)
                logger.info("Model SVM został wczytany.")
            except Exception as e:
                logger.error(f"Nie udało się wczytać modelu SVM: {e}")
        else:
            logger.error(f"Nieobsługiwany typ modelu: {model_type}")
            self.model = None

    def update_frame(self):
        if not self.camera_handler.camera_active:
            self.display_inactive_camera()
            return

        frame = self.camera_handler.get_frame()

        if len(frame) == 4:
            qimg, real_fps, fps, mp_results = frame
        else:
            qimg, real_fps, fps = frame
            mp_results = None

        if qimg is not None:
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
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

        if fps > 0:
            self.fps_declared_label.setText(f"Deklarowane FPS kamery: {real_fps:.2f}")
            self.fps_processing_label.setText(f"Przetwarzane FPS: {fps:.2f}")

    def display_inactive_camera(self):
        black_pixmap = QPixmap(self.video_label.width(), self.video_label.height())
        black_pixmap.fill(Qt.black)
        self.video_label.setPixmap(black_pixmap)
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
        except Exception as e:
            logger.error(f"Błąd podczas predykcji: {e}")
            self.predictionTextEdit.setPlainText("Predykcja: błąd")

    def update_prediction(self, new_prediction, max_proba):
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

    def update_accumulated_text(self, new_prediction):
        if new_prediction == "BACKSPACE":
            if self.accumulated_text:
                self.accumulated_text = self.accumulated_text[:-1]
        elif new_prediction == "SPACE":
            self.accumulated_text += " "
        else:
            self.accumulated_text += new_prediction
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

    def toggle_drawing(self):
        self.camera_handler.set_draw_landmarks(self.draw_checkbox.isChecked())

    def update_mediapipe_settings(self):
        try:
            model_complexity = int(self.model_complexity_combobox.currentText())
            det_conf = self.detection_spinbox.value()
            track_conf = self.tracking_spinbox.value()
            self.camera_handler.set_mediapipe_settings(model_complexity, det_conf, track_conf)
            logger.info("Ustawienia MediaPipe zostały zaktualizowane.")
        except Exception as e:
            logger.error(f"Nie udało się zaktualizować ustawień MediaPipe: {e}")

    def reset_models_settings(self):
        try:
            self.model_type_combobox.blockSignals(True)
            self.threshold_spinbox.blockSignals(True)
            self.stabilization_frames_spinbox.blockSignals(True)

            self.model_type_combobox.setCurrentIndex(0)
            self.threshold_spinbox.setValue(PREDICTION_THRESHOLD)
            self.stabilization_frames_spinbox.setValue(STABILIZATION_FRAMES)

            logger.info("Ustawienia modeli zostały zresetowane.")
        except Exception as e:
            logger.error(f"Nie udało się zresetować ustawień modeli: {e}")
        finally:
            self.model_type_combobox.blockSignals(False)
            self.threshold_spinbox.blockSignals(False)
            self.stabilization_frames_spinbox.blockSignals(False)

    def reset_mediapipe_settings(self):
        try:
            self.model_complexity_combobox.blockSignals(True)
            self.detection_spinbox.blockSignals(True)
            self.tracking_spinbox.blockSignals(True)
            self.draw_checkbox.blockSignals(True)

            self.model_complexity_combobox.setCurrentText("1")
            self.detection_spinbox.setValue(MEDIAPIPE_SETTINGS["min_detection_confidence"])
            self.tracking_spinbox.setValue(MEDIAPIPE_SETTINGS["min_tracking_confidence"])
            self.draw_checkbox.setChecked(True)

            logger.info("Ustawienia MediaPipe zostały zresetowane.")
        except Exception as e:
            logger.error(f"Nie udało się zresetować ustawień MediaPipe: {e}")
        finally:
            self.model_complexity_combobox.blockSignals(False)
            self.detection_spinbox.blockSignals(False)
            self.tracking_spinbox.blockSignals(False)
            self.draw_checkbox.blockSignals(False)

        self.update_mediapipe_settings()

    def update_prediction_threshold(self, new_value):
        self.prediction_threshold = new_value

    def update_stabilization_frames(self, new_value):
        self.stabilization_frames = new_value
        self.progress_bar.setRange(0, new_value)

    def closeEvent(self, event):
        self.camera_handler.close()
        event.accept()

    def detection_value_changed(self):
        self.update_mediapipe_settings()

    def tracking_value_changed(self):
        self.update_mediapipe_settings()

    def model_complexity_current_index_changed(self):
        self.model_complexity_combobox.clearFocus()
        self.update_mediapipe_settings()

    def detection_edit_finished(self):
        self.detection_spinbox.clearFocus()

    def tracking_edit_finished(self):
        self.tracking_spinbox.clearFocus()

    def clear_accumulated_text(self):
        self.accumulated_text = ""
        self.accumulatedTextEdit.clear()
