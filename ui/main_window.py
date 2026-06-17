from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QSizePolicy, QSpacerItem, QStatusBar
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 
from ui.quality_engineer_page     import QualityEngineerPage
from ui.quality_engineer_charts_page import QualityEngineerChartsPage
from ui.mathematical_specialist_page import MathematicalSpecialistPage
from ui.mathematical_specialist_charts_page import MathematicalSpecialistChartsPage

NAV_ITEMS_QUALITY = [
    ('Прогнозирование', 0, 'Прогнозирование'),
    ('Графики', 1, 'Графики'),
]

NAV_ITEMS_MATH = [
    ('Обучение', 0, 'Обучение'),
    ('Графики', 1, 'Графики'),
]

ROLE_DISPLAY = {
    "quality": "Инженер по качеству",
    "math":  "Специалист по мат. обеспечению",
}

class SideBar(QWidget):
    page_changed = Signal(int, str)

    def __init__(self, nav_items: list[tuple[str, int, str]],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('sidebar')
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._nav_buttons: list[QPushButton] = []
        self._build_ui(nav_items)

    def _build_ui(self, nav_items: list[tuple[str, int, str]]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("sidebarDivider")
        layout.addWidget(divider)
 
        layout.addSpacing(8)
 
        nav_lbl = QLabel("НАВИГАЦИЯ")
        nav_lbl.setObjectName("sectionLabel")
        nav_lbl.setContentsMargins(20, 0, 0, 0)
        layout.addWidget(nav_lbl)
        layout.addSpacing(4)

        for label, stack_index, page_title in nav_items:
            btn = QPushButton(label)
            btn.setObjectName('navBtn')
            btn.setCheckable(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(44)
            btn.clicked.connect(
                lambda checked, i=stack_index, t=page_title: self._on_nav(i, t)
            )
            self._nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        bottom_div = QFrame()
        bottom_div.setFrameShape(QFrame.Shape.HLine)
        bottom_div.setObjectName("sidebarDivider")
        layout.addWidget(bottom_div)
 
        # Activate first button by default
        if self._nav_buttons:
            self._set_active(0)

    def _on_nav(self, index: int, title: str) -> None:
        self._set_active(index)
        self.page_changed.emit(index, title)

    def _set_active(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty('active', i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)


class TopBar(QWidget):
    logout_requested = Signal()

    def __init__(self, username: str, role: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('topBar')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui(username, role)

    def _build_ui(self, username: str, role: str) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 16, 0)
        layout.setSpacing(12)

        self.page_title = QLabel('Рабочая область')
        self.page_title.setObjectName('topBarTitle')
        layout.addWidget(self.page_title)

        layout.addStretch()

        role_display = ROLE_DISPLAY.get(role, role)
        role_badge = QLabel(role_display.upper())
        role_badge.setObjectName('roleBadge')
        layout.addWidget(role_badge)

        user_lbl = QLabel(f'{username}')
        user_lbl.setObjectName('userBadge')
        layout.addWidget(user_lbl)

        logout_btn = QPushButton('Выйти')
        logout_btn.setObjectName('ghostBtn')
        logout_btn.setMinimumHeight(30)
        logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        logout_btn.clicked.connect(self.logout_requested)
        layout.addWidget(logout_btn)

    def set_title(self, title: str) -> None:
        self.page_title.setText(title)

class MainWindow(QMainWindow):
    logout_requested = Signal()

    def __init__(self, username: str, role: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.username = username
        self.role = role

        self.engine = create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/extcaland.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)

        self.setObjectName('mainWindow')
        self.setWindowTitle('Main Window')
        self.resize(1440, 900)
        self.setMinimumSize(1024, 700)

        self._build_ui()
        self._setup_status_bar()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName('centralWidget')
        central.setAutoFillBackground(True)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        nav_items, pages = self._build_role_content()

        self.top_bar = TopBar(self.username, self.role)
        self.top_bar.logout_requested.connect(self.logout_requested)
        main_layout.addWidget(self.top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.sidebar = SideBar(nav_items)
        self.sidebar.page_changed.connect(self._on_page_change)
        body.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setObjectName('pageStack')
        for page in pages:
            self.stack.addWidget(page)
        body.addWidget(self.stack, 1)

        main_layout.addLayout(body, 1)

        if nav_items:
            self.top_bar.set_title(nav_items[0][2])

    def _build_role_content(self):
        if self.role == 'math':
            work_page = MathematicalSpecialistPage(self.sessionmaker)
            charts_page = MathematicalSpecialistChartsPage()

            work_page.training_data_loaded.connect(
                charts_page.update_production_graph
            )

            work_page.training_complete.connect(
                charts_page.update_training_graph
            )

            pages=[work_page, charts_page]
            nav = NAV_ITEMS_MATH
        else:
            work_page = QualityEngineerPage(self.sessionmaker)
            charts_page = QualityEngineerChartsPage()

            work_page.production_data_loaded.connect(
                charts_page.update_production_graph
            )

            work_page.prediction_ready.connect(
                charts_page.update_prediction_graph
            )

            pages=[work_page, charts_page]
            nav = NAV_ITEMS_QUALITY

        return nav, pages

    def _setup_status_bar(self) -> None:
        sb = QStatusBar()
        sb.setObjectName('statusBar')
        role_txt = ROLE_DISPLAY.get(self.role, self.role)
        sb.showMessage(
            f'Пользователь: {self.username}'
            f' | Роль: {role_txt}'
        )

        self.setStatusBar(sb)

    def _on_page_change(self, index: int, title: str) -> None:
        self.stack.setCurrentIndex(index)
        self.top_bar.set_title(title)
