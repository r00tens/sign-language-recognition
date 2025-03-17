# DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50
LOG_LEVEL = 10

PREDICTION_THRESHOLD = 0.9
STABILIZATION_FRAMES = 20

CAMERA_CONFIG = {
    "index": 0,
    "width": 640,
    "height": 480,
    "video_codec": "MJPG",
    "fps": 30,
    "camera_active": True,
}

MEDIAPIPE_SETTINGS = {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

DRAWING_CONFIG = {
    "landmark_color": (235, 219, 178),
    "landmark_thickness": 1,
    "landmark_radius": 2,
    "connection_color": (235, 219, 178),
    "connection_thickness": 2,
}

SEED = 42
TRAINED_MODELS_DIR = "trained-models"
MODEL_SVM = "svm-C1.0-kernelrbf-degree3-gammascale-coef00.0.pkl"

LETTERS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "SPACE",
    27: "BACKSPACE",
}
