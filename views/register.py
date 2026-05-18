from PySide6.QtGui import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel, QVBoxLayout, QFrame
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from share import Page
from db.db_manager import user_exists, add_user

class RegisterView(QWidget):
    register_success = Signal()

    def __init__(self, nav):
        super().__init__()

        self.engine = sa.create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/users.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)

        self.nav = nav
        
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

        self.username = QLineEdit(placeholderText='Введите имя пользователя...')
        self.password = QLineEdit(placeholderText='Введите пароль...')
        self.password.setEchoMode(QLineEdit.EchoMode.Password)

        self.reg_btn = QPushButton('Зарегистрироваться')
        self.back_btn = QPushButton('Назад')

        self.reg_btn.clicked.connect(self.add_new_user)
        self.back_btn.clicked.connect(self.return_to_login)

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.username)
        frame_layout.addWidget(self.password)
        frame_layout.addWidget(self.reg_btn)
        frame_layout.addWidget(self.back_btn)

        layout.addWidget(frame)

        self.setLayout(layout)

    def add_new_user(self):
        username = self.username.text()
        password = self.password.text()

    def return_to_login(self):
        self.nav.navigate(Page.LOGIN)
