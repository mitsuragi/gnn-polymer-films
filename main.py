from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget)
from PySide6.QtCore import QFile, QTextStream
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import sys

from views import LoginView, RegisterView, MathSpecialistView, QualityEngineerView
from core.auth_service import AuthService
from core.navigation import NavigationManager
from share import Page

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() 

        self.setWindowTitle('GNN')
        self.setFixedSize(1600, 900)
        self.load_styles()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.nav = NavigationManager(self.stack)

        self.nav.register(Page.LOGIN, lambda nav: LoginView(nav))
        self.nav.register(Page.REGISTER, lambda nav: RegisterView(nav))
        self.nav.register(Page.MATH, lambda nav: MathSpecialistView(nav))
        self.nav.register(Page.QUALITY, lambda nav: QualityEngineerView(nav))

        self.nav.navigate(Page.LOGIN)

    def show_login(self):
        self.nav.navigate(Page.LOGIN)

    def show_register(self):
        self.nav.navigate(Page.REGISTER)

    def show_math_spec_view(self):
        self.nav.navigate(Page.MATH)

    def show_quality_eng_view(self):
        self.nav.navigate(Page.QUALITY)

    def on_login_success(self, user):
        if user.Role == 'math':
            self.show_math_spec_view()
        elif user.Role == 'quality':
            self.show_quality_eng_view()

    def load_styles(self):
        style_file = QFile("styles/styles.qss")
    
        if not style_file.exists():
            print('file not found')
            return
    
        if style_file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(style_file)
            self.setStyleSheet(stream.readAll())
            style_file.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
