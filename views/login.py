from PySide6.QtGui import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel, QVBoxLayout, QFrame

# from share.toast import ToastAlert

class LoginView(QWidget):
    login_success = Signal()
    show_register = Signal()

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Вход')
        # self.setFixedSize(512,512)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame = QFrame()
        frame.setFixedSize(300,300)

        frame_layout = QVBoxLayout(frame)

        self.label = QLabel('Вход')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.login = QLineEdit(placeholderText='Введите логин...')
        self.password = QLineEdit(placeholderText='Введите пароль...')

        self.log_btn = QPushButton('Войти')
        self.reg_btn = QPushButton('Зарегистрироваться')

        self.log_btn.clicked.connect(self.on_login)
        self.reg_btn.clicked.connect(self.show_register)

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.login)
        frame_layout.addWidget(self.password)
        frame_layout.addWidget(self.log_btn)
        frame_layout.addWidget(self.reg_btn)

        layout.addWidget(frame)

        self.setLayout(layout)

    def on_login(self):
        self.login_success.emit()
        # ToastAlert(self.window(), 'error', 'error')
