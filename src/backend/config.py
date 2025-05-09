from pathlib import Path
from typing import Optional

# DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50
LOG_LEVEL = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_THRESHOLD_SVM = 0.9
PREDICTION_THRESHOLD_CNN_TRANSFORMER = 0.5
STABILIZATION_FRAMES = 20  # for SVM
SEQUENCE_LENGTH = 40  # for CNN-Transformer


class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    video_codec: str = "MJPG"
    fps: int = 30
    camera_active: bool = True


class MediaPipeConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class DrawingConfig:
    landmark_color: tuple = (235, 219, 178)
    landmark_thickness: int = 1
    landmark_radius: int = 2
    connection_color: tuple = (235, 219, 178)
    connection_thickness: int = 2


class GestureConfig:
    num_classes: int = 250
    rows_per_frame: int = 543
    max_len: int = 384
    pad: float = -100.0


class TrainingConfig:
    batch_size: int = 64
    epochs: int = 250
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    dim: int = 144
    kernel_size: int = 19
    num_heads: int = 4
    expand: int = 2
    momentum: float = 0.99
    attn_dropout: float = 0.2
    drop_rate: float = 0.3
    label_smoothing: float = 0.05
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None
    augmentation: bool = True
    precision: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str = "GPU"  # "TPU", "GPU", "CPU"
    verbose: int = 2  # 0 silent, 1 progress bar, 2 one line per epoch
    model_name: str = "cnn-transformer"


SEED = 42
TRAINED_MODELS_DIR = "trained-models"
MODEL_SVM = "svm-C1.0-kernelrbf-degree3-gammascale-coef00.0.pkl"
MODEL_CNN_TRANSFORMER = "cnn-transformer.tflite"

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

NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]

LIP = [
    0,
    61,
    185,
    40,
    39,
    37,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]

REYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
]
LEYE = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]

import numpy as np

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(LANDMARKS)
CHANNELS = 6 * NUM_NODES

SELECTED_WORDS = {
    "TV": 0,
    # "after": 1,
    "airplane": 2,
    # "all": 3,
    # "alligator": 4,
    "animal": 5,
    # "another": 6,
    # "any": 7,
    "apple": 8,
    "arm": 9,
    "aunt": 10,
    # "awake": 11,
    # "backyard": 12,
    "bad": 13,
    # "balloon": 14,
    "bath": 15,
    "because": 16,
    "bed": 17,
    "bedroom": 18,
    # "bee": 19,
    # "before": 20,
    # "beside": 21,
    "better": 22,
    # "bird": 23,
    "black": 24,
    "blow": 25,
    "blue": 26,
    # "boat": 27,
    "book": 28,
    "boy": 29,
    "brother": 30,
    # "brown": 31,
    # "bug": 32,
    "bye": 33,
    # "callonphone": 34,
    "can": 35,
    "car": 36,
    # "carrot": 37,
    "cat": 38,
    # "cereal": 39,
    # "chair": 40,
    # "cheek": 41,
    "child": 42,
    # "chin": 43,
    # "chocolate": 44,
    # "clean": 45,
    # "close": 46,
    # "closet": 47,
    # "cloud": 48,
    # "clown": 49,
    # "cow": 50,
    # "cowboy": 51,
    "cry": 52,
    "cut": 53,
    # "cute": 54,
    "dad": 55,
    "dance": 56,
    "dirty": 57,
    "dog": 58,
    # "doll": 59,
    # "donkey": 60,
    # "down": 61,
    # "drawer": 62,
    "drink": 63,
    # "drop": 64,
    # "dry": 65,
    # "dryer": 66,
    # "duck": 67,
    "ear": 68,
    # "elephant": 69,
    # "empty": 70,
    # "every": 71,
    "eye": 72,
    # "face": 73,
    # "fall": 74,
    # "farm": 75,
    "fast": 76,
    # "feet": 77,
    "find": 78,
    # "fine": 79,
    # "finger": 80,
    "finish": 81,
    # "fireman": 82,
    "first": 83,
    # "fish": 84,
    # "flag": 85,
    # "flower": 86,
    "food": 87,
    "for": 88,
    # "frenchfries": 89,
    # "frog": 90,
    # "garbage": 91,
    # "gift": 92,
    # "giraffe": 93,
    "girl": 94,
    "give": 95,
    # "glasswindow": 96,
    "go": 97,
    # "goose": 98,
    # "grandma": 99,
    # "grandpa": 100,
    # "grass": 101,
    "green": 102,
    # "gum": 103,
    "hair": 104,
    "happy": 105,
    "hat": 106,
    # "hate": 107,
    "have": 108,
    "haveto": 109,
    "head": 110,
    # "hear": 111,
    # "helicopter": 112,
    "hello": 113,
    # "hen": 114,
    # "hesheit": 115,
    # "hide": 116,
    # "high": 117,
    "home": 118,
    # "horse": 119,
    "hot": 120,
    "hungry": 121,
    # "icecream": 122,
    "if": 123,
    # "into": 124,
    # "jacket": 125,
    # "jeans": 126,
    "jump": 127,
    # "kiss": 128,
    # "kitty": 129,
    # "lamp": 130,
    # "later": 131,
    "like": 132,
    # "lion": 133,
    "lips": 134,
    # "listen": 135,
    # "look": 136,
    # "loud": 137,
    "mad": 138,
    "make": 139,
    "man": 140,
    # "many": 141,
    "milk": 142,
    # "minemy": 143,
    # "mitten": 144,
    "mom": 145,
    # "moon": 146,
    "morning": 147,
    # "mouse": 148,
    # "mouth": 149,
    "nap": 150,
    # "napkin": 151,
    "night": 152,
    "no": 153,
    # "noisy": 154,
    "nose": 155,
    "not": 156,
    "now": 157,
    # "nuts": 158,
    "old": 159,
    "on": 160,
    # "open": 161,
    # "orange": 162,
    # "outside": 163,
    # "owie": 164,
    # "owl": 165,
    # "pajamas": 166,
    # "pen": 167,
    # "pencil": 168,
    # "penny": 169,
    "person": 170,
    # "pig": 171,
    # "pizza": 172,
    "please": 173,
    # "police": 174,
    # "pool": 175,
    # "potty": 176,
    # "pretend": 177,
    # "pretty": 178,
    # "puppy": 179,
    # "puzzle": 180,
    # "quiet": 181,
    # "radio": 182,
    # "rain": 183,
    "read": 184,
    "red": 185,
    # "refrigerator": 186,
    # "ride": 187,
    # "room": 188,
    "sad": 189,
    # "same": 190,
    # "say": 191,
    "scissors": 192,
    # "see": 193,
    "shhh": 194,
    # "shirt": 195,
    "shoe": 196,
    "shower": 197,
    "sick": 198,
    "sleep": 199,
    # "sleepy": 200,
    "smile": 201,
    # "snack": 202,
    # "snow": 203,
    # "stairs": 204,
    # "stay": 205,
    # "sticky": 206,
    # "store": 207,
    # "story": 208,
    # "stuck": 209,
    # "sun": 210,
    # "table": 211,
    "talk": 212,
    # "taste": 213,
    "thankyou": 214,
    "that": 215,
    # "there": 216,
    "think": 217,
    # "thirsty": 218,
    # "tiger": 219,
    "time": 220,
    "tomorrow": 221,
    # "tongue": 222,
    # "tooth": 223,
    # "toothbrush": 224,
    # "touch": 225,
    # "toy": 226,
    "tree": 227,
    "uncle": 228,
    # "underwear": 229,
    # "up": 230,
    # "vacuum": 231,
    "wait": 232,
    "wake": 233,
    "water": 234,
    # "wet": 235,
    # "weus": 236,
    "where": 237,
    "white": 238,
    "who": 239,
    "why": 240,
    "will": 241,
    # "wolf": 242,
    # "yellow": 243,
    "yes": 244,
    "yesterday": 245,
    # "yourself": 246,
    # "yucky": 247,
    # "zebra": 248,
    # "zipper": 249,
}

ALLOWED_LABELS = list(SELECTED_WORDS.values())
NUM_SELECTED = len(ALLOWED_LABELS)

NEW_MAPPING = {word: idx for idx, word in enumerate(SELECTED_WORDS.keys())}
INVERTED_NEW_MAPPING = {idx: word for word, idx in NEW_MAPPING.items()}
