from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QFrame, QTableWidget, QHeaderView, QTableWidgetItem
)
from PySide6.QtCore import Qt
import pandas as pd
 
from gnn.metrics_logger import MetricsLogger
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
            'Динамика метрики precision на обучающей и валидационной выборках по эпохам'
        )
        desc1.setObjectName('sectionLabel')
        desc1.setWordWrap(True)
        tab1_layout.addWidget(desc1)

        self.precision_chart = MatplotlibWidget()
        self.precision_chart.setObjectName('msTrainingChartFull')
        tab1_layout.addWidget(self.precision_chart, 1)
        self.tab_widget.addTab(tab1, 'Precision')

        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 8, 0, 0)

        desc2 = QLabel(
            'Динамика метрики recall на обучающей и валидационной выборках по эпохам'
        )
        desc2.setObjectName('sectionLabel')
        desc2.setWordWrap(True)
        tab2_layout.addWidget(desc2)

        self.recall_chart = MatplotlibWidget()
        self.recall_chart.setObjectName('msTrainingChartFull')
        tab2_layout.addWidget(self.recall_chart, 1)
        self.tab_widget.addTab(tab2, 'Recall')

        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(0, 8, 0, 0)

        desc3 = QLabel(
            'Динамика метрики f1-score на обучающей и валидационной выборках по эпохам'
        )
        desc3.setObjectName('sectionLabel')
        desc3.setWordWrap(True)
        tab3_layout.addWidget(desc3)

        self.f1_chart = MatplotlibWidget()
        self.f1_chart.setObjectName('msTrainingChartFull')
        tab3_layout.addWidget(self.f1_chart, 1)
        self.tab_widget.addTab(tab3, 'F1-score')

        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(0, 8, 0, 0)

        desc4 = QLabel(
            'Динамика метрики PR-AUC на обучающей и валидационной выборках по эпохам'
        )
        desc4.setObjectName('sectionLabel')
        desc4.setWordWrap(True)
        tab4_layout.addWidget(desc4)

        self.pr_auc_chart = MatplotlibWidget()
        self.pr_auc_chart.setObjectName('msTrainingChartFull')
        tab4_layout.addWidget(self.pr_auc_chart, 1)
        self.tab_widget.addTab(tab4, 'PR-AUC')

        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        tab5_layout.setContentsMargins(0, 8, 0, 0)

        desc5 = QLabel(
            'Таблица метрик обучающей и валидационной выборках по эпохам'
        )
        desc5.setObjectName('sectionLabel')
        desc5.setWordWrap(True)
        tab5_layout.addWidget(desc5)

        self.metrics_table = QTableWidget()
        self.metrics_table.setObjectName('metricsModelTable')
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels([
            'Эпоха', 'Этап', 'Precision', 'Recall', 'F1-score', 'PR-AUC'
        ])
        self.metrics_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection)
        self.metrics_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.metrics_table.setSortingEnabled(False)
        self.metrics_table.setAlternatingRowColors(False)
        self.metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        tab5_layout.addWidget(self.metrics_table, 1)
        self.tab_widget.addTab(tab5, 'Таблица метрик')

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
        self.production_chart.setObjectName('msProductionChartFull')
        tab6_layout.addWidget(self.production_chart, 1)
        self.tab_widget.addTab(tab6, 'Производственные данные')

        root.addWidget(self.tab_widget, 1)

    def _load_train_data(self, source: MetricsLogger | str) -> pd.DataFrame:
        if isinstance(source, MetricsLogger):
            return pd.DataFrame([
                {"epoch": m.epoch, "phase": m.phase,
                 "precision": m.precision, "recall": m.recall, "f1": m.f1, "pr-auc": m.pr_auc}
                for m in source.history
            ])
        return pd.read_csv(source)[["epoch", "phase", "precision", "recall", "f1", "pr-auc"]]

    def _fill_metrics_table(self, df: pd.DataFrame) -> None:
        self.metrics_table.setRowCount(len(df))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            self.metrics_table.setItem(
                row_idx, 0, QTableWidgetItem(str(row["epoch"]))
            )
            self.metrics_table.setItem(
                row_idx, 1, QTableWidgetItem(str(row["phase"]))
            )
            self.metrics_table.setItem(
                row_idx, 2, QTableWidgetItem(f"{row['precision']:.4f}")
            )
            self.metrics_table.setItem(
                row_idx, 3, QTableWidgetItem(f"{row['recall']:.4f}")
            )
            self.metrics_table.setItem(
                row_idx, 4, QTableWidgetItem(f"{row['f1']:.4f}")
            )
            self.metrics_table.setItem(
                row_idx, 5, QTableWidgetItem(f"{row['pr-auc']:.4f}")
            )

    def update_training_graph(self, logger) -> None:
        df = self._load_train_data(logger)

        self.precision_chart.plot_metric(df, 'precision')
        self.recall_chart.plot_metric(df, 'recall')
        self.f1_chart.plot_metric(df, 'f1')
        self.pr_auc_chart.plot_metric(df, 'pr-auc')
        self._fill_metrics_table(df)

    def update_production_graph(self, time, values) -> None:
        self.production_chart.plot_demo_production(time, values)
