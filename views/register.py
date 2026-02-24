from PySide6.QtGui import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QPushButton, QTextEdit, QLabel, QVBoxLayout, QFrame

class RegisterView(QWidget):
    register_success = Signal()

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Регистрация')
        # self.setFixedSize(512,512)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame = QFrame()
        frame.setFixedSize(300,300)

        frame_layout = QVBoxLayout(frame)

        self.label = QLabel('Регистрация')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.login = QTextEdit(placeholderText='Введите логин...')
        self.password = QTextEdit(placeholderText='Введите пароль...')

        self.reg_btn = QPushButton('Зарегистрироваться')
        self.back_btn = QPushButton('Назад')

        self.reg_btn.clicked.connect(self.add_new_user)
        self.back_btn.clicked.connect(self.return_to_login)

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.login)
        frame_layout.addWidget(self.password)
        frame_layout.addWidget(self.reg_btn)
        frame_layout.addWidget(self.back_btn)

        layout.addWidget(frame)

        self.setLayout(layout)

    def add_new_user(self):
        self.register_success.emit()

    def return_to_login(self):
        self.register_success.emit()
