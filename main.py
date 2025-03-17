import sys

from PySide6.QtWidgets import QApplication

from src.backend.themes.pallets import gruvbox_palette
from src.frontend.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(gruvbox_palette())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
