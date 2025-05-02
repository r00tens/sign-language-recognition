import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

from src.backend.config import LOG_LEVEL, PROJECT_ROOT
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


class SVM:
    def __init__(self, random_state: int = 42):
        self.random_state: int = random_state
        self.model: Optional[SVC] = None
        self.best_params: Optional[dict] = None

    def initialize_model(self) -> None:
        defaults = {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "shrinking": True,
            "probability": True,
            "tol": 1e-3,
            "cache_size": 8192,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": "ovr",
            "break_ties": False,
            "random_state": self.random_state,
        }

        if self.best_params is not None:
            logger.info(f"Trenowanie modelu z najlepszymi hiperparametrami: {self.best_params}")
            defaults.update(self.best_params)
        else:
            logger.info("Trenowanie modelu z domyślnymi hiperparametrami.")

        self.model = SVC(**defaults)

    def _ensure_model_initialized(self) -> SVC:
        if self.model is None:
            raise ValueError("Model nie został zainicjalizowany.")

        return self.model

    def _ensure_model_trained(self) -> SVC:
        model = self._ensure_model_initialized()
        if not hasattr(model, "support_vectors_"):
            raise ValueError("Model nie został wytrenowany.")

        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._ensure_model_initialized().fit(x, y)

        logger.info("Model został wytrenowany.")

    def perform_cross_validation(self, x: np.ndarray, y: np.ndarray, cv: int) -> np.ndarray:
        model = self._ensure_model_initialized()
        cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)

        logger.info(f"Wyniki walidacji krzyżowej: {cv_scores}")
        logger.info(f"Średnia dokładność walidacji krzyżowej: {np.mean(cv_scores)}")

        return cv_scores

    def tuning_hyperparameters(self, x: np.ndarray, y: np.ndarray, cv: int = 5) -> Any:
        model = self._ensure_model_initialized()
        param_grid = [
            {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
            {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01]},
            {
                "kernel": ["poly"],
                "C": [0.1, 1, 10, 100],
                "gamma": [1, 0.1, 0.01, 0.001],
                "degree": [2, 3, 4],
                "coef0": [0, 0.1, 0.5, 1],
            },
            {
                "kernel": ["sigmoid"],
                "C": [0.1, 1, 10, 100],
                "gamma": [1, 0.1, 0.01, 0.001],
                "coef0": [0, 0.1, 0.5, 1],
            },
        ]
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(x, y)
        self.best_params = grid_search.best_params_

        logger.info(f"Najlepsze hiperparametry: {self.best_params}")

        return grid_search.cv_results_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._ensure_model_trained().predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._ensure_model_trained().predict_proba(x)

    def save_model(self, filepath: Path) -> None:
        model = self._ensure_model_trained()
        with filepath.open("wb") as f:
            pickle.dump(model, f)  # type: ignore

    def load_model(self, filepath: Path) -> None:
        with filepath.open("rb") as f:
            self.model = pickle.load(f)

    def evaluate(self, x: np.ndarray, y: np.ndarray, plot_confusion: bool = False) -> dict:
        model = self._ensure_model_trained()
        y_pred = model.predict(x)
        report = classification_report(y, y_pred, digits=4)
        matrix = confusion_matrix(y, y_pred)
        accuracy = accuracy_score(y, y_pred)

        logger.info(f"Raport klasyfikacji:\n{report}")
        logger.info(f"Dokładność: {accuracy}")

        if plot_confusion:
            plt.figure(figsize=(8, 6))
            cmap = sns.light_palette("gold", as_cmap=True)
            sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, square=True)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / "data/training-results/svm-confusion_matrix.png")

        return {
            "classification_report": report,
            "confusion_matrix": matrix,
            "accuracy": accuracy,
        }
