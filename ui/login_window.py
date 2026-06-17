from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon
 
from ui.register_dialog import RegisterDialog

class LoginWindow(QWidget):
    login_requested = Signal(str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('loginBackground')
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
 
        # Centered card
        card = QFrame()
        card.setObjectName("loginContainer")
        card.setFixedWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 36, 40, 36)
        card_layout.setSpacing(14)
 
        # ── Branding ─────────────────────────────────────────────────────────
        brand_box = QVBoxLayout()
        brand_box.setSpacing(4)
 
        title = QLabel("GNN")
        title.setObjectName("loginTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
 
        brand_box.addWidget(title)
        card_layout.addLayout(brand_box)
 
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("sidebarDivider")
        card_layout.addWidget(divider)
        card_layout.addSpacing(6)
 
        # ── Login label ───────────────────────────────────────────────────────
        login_lbl = QLabel("ВХОД В СИСТЕМУ")
        login_lbl.setObjectName("sectionLabel")
        login_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(login_lbl)
        card_layout.addSpacing(4)
 
        # ── Username ──────────────────────────────────────────────────────────
        user_lbl = QLabel("Логин")
        self.username_input = QLineEdit()
        self.username_input.setObjectName("usernameInput")
        self.username_input.setPlaceholderText("Введите логин...")
        self.username_input.setMinimumHeight(38)
        card_layout.addWidget(user_lbl)
        card_layout.addWidget(self.username_input)
 
        # ── Password ──────────────────────────────────────────────────────────
        pass_lbl = QLabel("Пароль")
        self.password_input = QLineEdit()
        self.password_input.setObjectName("passwordInput")
        self.password_input.setPlaceholderText("Введите пароль...")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setMinimumHeight(38)
        self.password_input.returnPressed.connect(self._on_login)
        card_layout.addWidget(pass_lbl)
        card_layout.addWidget(self.password_input)
 
        card_layout.addSpacing(8)
 
        # ── Primary action ────────────────────────────────────────────────────
        self.login_button = QPushButton("Войти")
        self.login_button.setObjectName("primaryButton")
        self.login_button.setMinimumHeight(42)
        self.login_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_button.clicked.connect(self._on_login)
        card_layout.addWidget(self.login_button)
 
        # ── Secondary action ──────────────────────────────────────────────────
        self.register_button = QPushButton("Создать пользователя")
        self.register_button.setObjectName("ghostButton")
        self.register_button.setMinimumHeight(36)
        self.register_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.register_button.clicked.connect(self._open_register_dialog)
        card_layout.addWidget(self.register_button)
 
        # ── Status feedback ───────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        card_layout.addWidget(self.status_label)
 
        root.addWidget(card)

    def _on_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            self.show_error('Заполните все поля')
            return 

        self.login_requested.emit(username, password)

    def _open_register_dialog(self) -> None:
        dlg = RegisterDialog(self)
        dlg.exec()

    def show_error(self, message: str) -> None:
        self.status_label.setStyleSheet("color: #FF6B6B;")
        self.status_label.setText(message)
 
    def show_success(self, message: str) -> None:
        self.status_label.setStyleSheet("color: #7AE08A;")
        self.status_label.setText(message)
 
    def clear_status(self) -> None:
        self.status_label.setText("")
