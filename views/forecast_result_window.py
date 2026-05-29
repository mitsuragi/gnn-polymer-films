from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg
from share import PredictionChartWidget

class ForecastResultWindow(QWidget):
    def __init__(self, result, df):
        super().__init__()
        self.setWindowTitle('Результаты прогнозирования')

        layout = QVBoxLayout(self)

        chart = PredictionChartWidget()
        chart.load(result=result, df=df, target_col='target')

        layout.addWidget(chart)
