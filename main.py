from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget)

import sys

from views import LoginView, RegisterView, MathSpecialistView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() 

        self.setWindowTitle('GNN')
        self.setFixedSize(1600, 900)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_view = LoginView()
        self.register_view = RegisterView()
        self.math_spec_view = MathSpecialistView()

        self.login_view.show_register.connect(self.show_register)
        self.login_view.login_success.connect(self.show_main)

        self.register_view.register_success.connect(self.show_login)

        self.stack.addWidget(self.math_spec_view)
        self.stack.addWidget(self.login_view)
        self.stack.addWidget(self.register_view)

    def show_login(self):
        self.stack.setCurrentWidget(self.login_view)

    def show_register(self):
        self.stack.setCurrentWidget(self.register_view)

    def show_main(self):
        self.stack.setCurrentWidget(self.math_spec_view)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
