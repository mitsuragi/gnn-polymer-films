from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFrame
)
from PySide6.QtCore import Qt, Signal

class RegisterDialog(QDialog):
    register_requested = Signal(str, str, str)

    ROLES = {
        'Инженер по качеству': 'quality_engineer',
        'Специалист по математическому обеспечению': 'math_specialist'
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('registerDialog')
        self.setWindowTitle('Регистрация')
        self.setModal(True)
        self.setFixedWidth(440)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)

        title = QLabel('Создание пользователя')
        title.setObjectName('sectionLabel')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName('sidebarDivider')
        layout.addWidget(divider)
        layout.addSpacing(4)

        layout.addWidget(QLabel('Логин'))
        self.username_input = QLineEdit()
        self.username_input.setObjectName('regUsernameInput')
        self.username_input.setPlaceholderText('Логин...')
        self.username_input.setMinimumHeight(36)
        layout.addWidget(self.username_input)

        layout.addWidget(QLabel('Пароль'))
        self.password_input = QLineEdit()
        self.password_input.setObjectName('regPasswordInput')
        self.password_input.setPlaceholderText('Пароль...')
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setMinimumHeight(36)
        layout.addWidget(self.password_input)

        layout.addWidget(QLabel('Подтверждение пароля'))
        self.confirm_input = QLineEdit()
        self.confirm_input.setObjectName('regConfirmInput')
        self.confirm_input.setPlaceholderText('Повторите пароль...')
        self.confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_input.setMinimumHeight(36)
        layout.addWidget(self.confirm_input)

        layout.addWidget(QLabel('Роль'))
        self.role_combo = QComboBox()
        self.role_combo.setObjectName('roleCombo')
        self.role_combo.setMinimumHeight(36)
        for display_name in self.ROLES:
            self.role_combo.addItem(display_name)
        layout.addWidget(self.role_combo)

        layout.addSpacing(8)

        self.status_label = QLabel('')
        self.status_label.setObjectName('statusLabel')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.cancel_btn = QPushButton('Отмена')
        self.cancel_btn.setObjectName('ghostBtn')
        self.cancel_btn.setMinimumHeight(38)
        self.cancel_btn.clicked.connect(self.reject)

        self.create_btn = QPushButton('Создать')
        self.create_btn.setObjectName('primaryBtn')
        self.create_btn.setMinimumHeight(38)
        self.create_btn.clicked.connect(self._on_create)

        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.create_btn)
        layout.addLayout(btn_row)

    def _on_create(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        role_display = self.role_combo.currentText()
        role_key = self.ROLES[role_display]

        if not username:
            self._show_error('Введите логин')
            return
        if password != confirm:
            self._show_error('Пароли не совпадают')
            return

        self.register_requested.emit(username, password, role_key)
        self._show_success(f'Пользователь {username} создан')

    def _show_error(self, msg: str) -> None:
        self.status_label.setStyleSheet('color: #FF6B6B;')
        self.status_label.setText(msg)

    def _show_success(self, msg: str) -> None:
        self.status_label.setStyleSheet('color: #7AE08A;')
        self.status_label.setText(msg)
        self.create_btn.setEnabled(False)
