from PySide6.QtGui import QFont, QFontDatabase

from src.backend.config import LOG_LEVEL
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def load_font(font_path: str, font_size: int = 10) -> QFont:
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        logger.error(f"Nie udało się załadować czcionki z {font_path}.")

        return QFont()
    else:
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            logger.info(
                f"Załadowano rodzinę czcionek: {font_families[0]} o rozmiarze {font_size}."
            )

            return QFont(font_families[0], font_size)
        else:
            logger.error(f"Nie udało się załadować czcionki z {font_path}.")

            return QFont()
