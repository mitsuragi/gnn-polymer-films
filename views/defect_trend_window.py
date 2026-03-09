from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg
import datetime

class DefectTrendWindow(QWidget):
    def __init__(self, times, values):
        super().__init__()
        self.setWindowTitle('Тренды значений дефектов')

        layout = QVBoxLayout(self)

        axis = pg.DateAxisItem(orientation='bottom')

        self.plot_widget = pg.PlotWidget(axisItems={'bottom': axis})

        layout.addWidget(self.plot_widget)

        x = [
            t.timestamp() for t in times
        ]

        self.plot_widget.plot(x, values, pen='b')

        self.plot_widget.showGrid(x=True, y=True)
