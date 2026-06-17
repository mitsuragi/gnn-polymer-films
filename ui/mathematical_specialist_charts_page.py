from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QPushButton, QGroupBox, QFrame
)
from PySide6.QtCore import Qt
 
from ui.widgets.matplotlib_widget import MatplotlibWidget

class MathematicalSpecialistChartsPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('mathChartsPage')
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel('Графики')
        title.setObjectName('topBarTitle')
        header.addWidget(title)
        header.addStretch()

        root.addLayout(header)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setObjectName('sidebarDivider')
        root.addWidget(div)

        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName('mathChartsTab')

        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(0, 8, 0, 0)

        desc1 = QLabel(
            'Динамика функции потерь на обучающей и валидационной выборках по эпохам'
        )
        desc1.setObjectName('sectionLabel')
        desc1.setWordWrap(True)
        tab1_layout.addWidget(desc1)

        self.training_chart = MatplotlibWidget()
        self.training_chart.setObjectName('msTrainingChartFull')
        tab1_layout.addWidget(self.training_chart, 1)
        self.tab_widget.addTab(tab1, 'График обучения')

        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 8, 0, 0)

        desc2 = QLabel(
            'Временной ряд значений выбранного показателя качества'
        )
        desc2.setObjectName('sectionLabel')
        desc2.setWordWrap(True)
        tab2_layout.addWidget(desc2)

        self.production_chart = MatplotlibWidget()
        self.production_chart.setObjectName('msProductionChartFull')
        tab2_layout.addWidget(self.production_chart, 1)
        self.tab_widget.addTab(tab2, 'Производственные данные')

        root.addWidget(self.tab_widget, 1)

    def update_training_graph(self, actual=None, predicted=None) -> None:
        self.training_chart.plot_demo_training()

    def update_production_graph(self, time, values) -> None:
        self.production_chart.plot_demo_production(time, values)
