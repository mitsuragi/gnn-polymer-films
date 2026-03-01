from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QRect
from PySide6.QtGui import QFont

class ToastAlert(QWidget):
    COLORS = {
        "success": "#4caf50",
        "warn":    "#ff9800",
        "error":   "#f44336"
    }

    BG = {
        "success": "#e8f5e9",
        "warn":    "#fff3e0",
        "error":   "#ffebee"
    }

    ICON = {
        "success": "✔",
        "warn":    "⚠",
        "error":   "✖"
    }

    def __init__(self, parent, type_: str, message: str, duration=3000):
        super().__init__(parent)

        self.setWindowFlags(
            Qt.WindowType.ToolTip |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating) 
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 24, 12) 
        layout.setSpacing(12)


        icon_lbl = QLabel(self.ICON.get(type_, "i"))
        icon_lbl.setFont(QFont("Segoe UI Emoji", 18))
        icon_lbl.setStyleSheet(f"color: {self.COLORS.get(type_, '#2196f3')};")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        text_lbl = QLabel(message)
        text_lbl.setStyleSheet("color: #212121; font-size: 14px; font-weight: 500;")
        text_lbl.setWordWrap(True)
        layout.addWidget(text_lbl, stretch=1)
        self.setStyleSheet(f"""
            QWidget {{
                background: {self.BG.get(type_, '#e3f2fd')};
                border-left: 5px solid {self.COLORS.get(type_, '#2196f3')};
                border-radius: 8px;
                padding: 4px;
            }}
        """)

    
        self.adjustSize()
        self.show()                  
        self.adjustSize()              

        main_window = parent.window()
        geo = main_window.geometry()
        
        margin_x = 10
        margin_y = 20 

        x = geo.right() - self.width() - margin_x
        y = geo.top() + margin_y

        self.move(x, y)

        self.setWindowOpacity(0.0)
        self._fade_in()

        QTimer.singleShot(duration, self._fade_out)
    
    def _fade_in(self):
        self.show()
        self.setWindowOpacity(0)

        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(250)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.start()

    def _fade_out(self):
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(1)
        self.anim.setEndValue(0)
        self.anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self.anim.finished.connect(self.close)
        self.anim.start()
