from PySide6.QtGui import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel, QVBoxLayout, QFrame
import sqlalchemy as sa 
from sqlalchemy.orm import sessionmaker

from share import Page
from core import AuthService

class LoginView(QWidget):
    login_success = Signal(object)
    login_failed = Signal(str)

    def __init__(self, nav):
        super().__init__()

        self.engine = sa.create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/users.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)

        self.nav = nav
        self.auth_service = AuthService(self.sessionmaker)
        
        self.setWindowTitle('Вход')

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame = QFrame()
        frame.setFixedSize(300,300)

        frame_layout = QVBoxLayout(frame)

        self.label = QLabel('Вход')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.username = QLineEdit(placeholderText='Введите имя пользователя...')
        self.password = QLineEdit(placeholderText='Введите пароль...')
        self.password.setEchoMode(QLineEdit.EchoMode.Password)

        self.log_btn = QPushButton('Войти')
        self.reg_btn = QPushButton('Зарегистрироваться')

        self.log_btn.clicked.connect(self.on_login)
        self.reg_btn.clicked.connect(self.to_register)

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.username)
        frame_layout.addWidget(self.password)
        frame_layout.addWidget(self.log_btn)
        frame_layout.addWidget(self.reg_btn)

        layout.addWidget(frame)

        self.setLayout(layout)

    def on_login(self):
        username = self.username.text()
        password = self.password.text()

        if self.auth_service is not None:
            user = self.auth_service.login(username, password)

            if not user:
                print('ошибка')
                return

            if user.Role == 'math':
                self.nav.navigate(Page.MATH)
            elif user.Role == 'quality':
                self.nav.navigate(Page.QUALITY)

    def to_register(self):
        self.nav.navigate(Page.REGISTER)
