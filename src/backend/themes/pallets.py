from PySide6.QtGui import QColor, QPalette


def gruvbox_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#282828"))
    palette.setColor(QPalette.WindowText, QColor("#ebdbb2"))
    palette.setColor(QPalette.Base, QColor("#3c3836"))
    palette.setColor(QPalette.AlternateBase, QColor("#504945"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ebdbb2"))
    palette.setColor(QPalette.ToolTipText, QColor("#282828"))
    palette.setColor(QPalette.Text, QColor("#ebdbb2"))
    palette.setColor(QPalette.Button, QColor("#282828"))
    palette.setColor(QPalette.ButtonText, QColor("#ebdbb2"))
    palette.setColor(QPalette.BrightText, QColor("#fb4934"))
    palette.setColor(QPalette.Link, QColor("#83a598"))
    palette.setColor(QPalette.Highlight, QColor("#d79921"))
    palette.setColor(QPalette.HighlightedText, QColor("#282828"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#7c6f64"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#7c6f64"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7c6f64"))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#3c3836"))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor("#7c6f64"))

    return palette
