from PySide6.QtGui import QFont, QFontDatabase

from src.backend.config import LOG_LEVEL
from src.backend.utils.app_logger import AppLogger

logger = AppLogger(name=__name__, level=LOG_LEVEL)


def load_font(font_path: str, font_size: int = 10) -> QFont:
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        logger.error(f"Failed to load font from {font_path}.")

        return QFont()
    else:
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            logger.info(f"Loaded font family: {font_families[0]} with size {font_size}.")

            return QFont(font_families[0], font_size)
        else:
            logger.error(f"Failed to load font from {font_path}.")

            return QFont()
