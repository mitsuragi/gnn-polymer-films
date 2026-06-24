from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QFrame
)
 
from ui.widgets.matplotlib_widget import MatplotlibWidget

class QualityEngineerChartsPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('qualityChartsPage')
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
        self.tab_widget.setObjectName('qualityChartsTab')

        # tab1 = QWidget()
        # tab1_layout = QVBoxLayout(tab1)
        # tab1_layout.setContentsMargins(0, 0, 0, 0)
        #
        # desc1 = QLabel(
        #     'Сравнение фактических и спрогнозированных значений показателя качества за выбранный период'
        # )
        # desc1.setObjectName('sectionLabel')
        # desc1.setWordWrap(True)
        # tab1_layout.addWidget(desc1)
        #
        # self.prediction_chart = MatplotlibWidget()
        # self.prediction_chart.setObjectName('qePredictionChartFull')
        # tab1_layout.addWidget(self.prediction_chart, 1)
        # self.tab_widget.addTab(tab1, 'Фактические и спрогнозированные значения')

        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 0, 0, 0)

        desc2 = QLabel(
            'Распределение ответов модели'
        )
        desc2.setObjectName('sectionLabel')
        desc2.setWordWrap(True)
        tab2_layout.addWidget(desc2)

        self.bar_chart = MatplotlibWidget()
        self.bar_chart.setObjectName('qePredictionChartFull')
        tab2_layout.addWidget(self.bar_chart, 1)
        self.tab_widget.addTab(tab2, 'Распределение ответов модели')

        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(0, 0, 0, 0)

        desc3 = QLabel(
            'Precision-Recall кривая'
        )
        desc3.setObjectName('sectionLabel')
        desc3.setWordWrap(True)
        tab3_layout.addWidget(desc3)

        self.auc_chart = MatplotlibWidget()
        self.auc_chart.setObjectName('qePredictionChartFull')
        tab3_layout.addWidget(self.auc_chart, 1)
        self.tab_widget.addTab(tab3, 'PR-AUC')

        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(0, 0, 0, 0)

        desc4 = QLabel(
            'Распределение предсказанных классов по вероятностям'
        )
        desc4.setObjectName('sectionLabel')
        desc4.setWordWrap(True)
        tab4_layout.addWidget(desc4)

        self.distribution_chart = MatplotlibWidget()
        self.distribution_chart.setObjectName('qePredictionChartFull')
        tab4_layout.addWidget(self.distribution_chart, 1)
        self.tab_widget.addTab(tab4, 'Распределение вероятностей')

        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        tab5_layout.setContentsMargins(0, 0, 0, 0)

        desc5 = QLabel(
            'График действительных и спрогнозированных значений'
        )
        desc5.setObjectName('sectionLabel')
        desc5.setWordWrap(True)
        tab5_layout.addWidget(desc5)

        self.prediction_chart = MatplotlibWidget()
        self.prediction_chart.setObjectName('qePredictionChartFull')
        tab5_layout.addWidget(self.prediction_chart, 1)
        self.tab_widget.addTab(tab5, 'График прогноза')

        tab6 = QWidget()
        tab6_layout = QVBoxLayout(tab6)
        tab6_layout.setContentsMargins(0, 8, 0, 0)

        desc6 = QLabel(
            'Временной ряд значений выбранного показателя качества'
        )
        desc6.setObjectName('sectionLabel')
        desc6.setWordWrap(True)
        tab6_layout.addWidget(desc6)

        self.production_chart = MatplotlibWidget()
        self.production_chart.setObjectName('qeProductionChartFull')
        tab6_layout.addWidget(self.production_chart, 1)
        self.tab_widget.addTab(tab6, 'Производственные данные')

        root.addWidget(self.tab_widget, 1)

    def update_prediction_graph(self, time, y_true, y_prob, y_pred, answer_distribution, threshold) -> None:

        self.bar_chart.plot_bar_chart(answer_distribution)
        self.auc_chart.plot_pr_auc_chart(y_true, y_prob)
        self.distribution_chart.plot_class_distribution(y_true, y_prob, threshold)
        self.prediction_chart.plot_demo_prediction(time, y_true, y_pred)

    def update_production_graph(self, time, values) -> None:
        self.production_chart.plot_demo_production(time, values)
