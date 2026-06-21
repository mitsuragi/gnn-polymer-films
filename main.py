import sys
import os
 
from PySide6.QtWidgets import QApplication
from PySide6.QtGui     import QFont
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
 
from ui.login_window import LoginWindow
from ui.main_window  import MainWindow
from core.auth_service import AuthService

class AppController:
    def __init__(self, app: QApplication):
        self.engine = sa.create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/users.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)

        self._app = app
        self._auth = AuthService(self.sessionmaker)
        self._login_window: LoginWindow | None = None
        self._main_win: MainWindow | None = None

    def start(self) -> None:
        self._show_login()

    def _show_login(self) -> None:
        if self._main_win:
            self._main_win.close()
            self._main_win = None

        self._login_window = LoginWindow()
        self._login_window.login_requested.connect(self._on_login)
        self._login_window.setWindowTitle('GNN')
        self._login_window.resize(900,600)
        self._login_window.setMinimumSize(700,500)
        self._login_window.show()

    def _on_login(self, username: str, password: str) -> None:
        result = self._auth.authenticate(username, password)

        if result:
            self._login_window.hide()
            self._open_main_window(result.Username, result.Role)
        else:
            self._login_window.show_error('Неверный логин или пароль')

    def _open_main_window(self, username: str, role: str) -> None:
        self._main_win = MainWindow(username, role)
        self._main_win.logout_requested.connect(self._on_logout)
        self._main_win.show()

    def _on_logout(self) -> None:
        self._show_login()

def load_styles(app: QApplication) -> None:
    qss_path = os.path.join(os.path.dirname(__file__), 'styles', 'style.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
    else:
        print(f'[Warning] Stylesheet not found: {qss_path}')

def main() -> int:
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    
    app = QApplication(sys.argv)
    app.setApplicationName('GNN')

    font = QFont('Segoe UI', 10)
    app.setFont(font)

    load_styles(app)

    controller = AppController(app)
    controller.start()

    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
