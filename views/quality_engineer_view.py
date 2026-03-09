from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QGridLayout, QHBoxLayout, QLabel, QListView, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, QFrame, QPushButton)
from PySide6.QtCore import Signal
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from share import AdjustedComboBox
from db.db_manager import get_datetime_range, get_models, get_model_data, get_forecasting_data
from .defect_trend_window import DefectTrendWindow

class QualityEngineerView(QWidget):
    quit_view_signal = Signal()
    model = None
    df = None

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Интерфейс инженера по качеству')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(15, 10, 15, 10)
        frame_layout.setSpacing(20)

        self.quit_view_btn = QPushButton('Выйти')
        self.quit_view_btn.clicked.connect(self.quit_view)

        header_layout = QGridLayout()
        header_layout.addWidget(self.quit_view_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(QLabel('Интерфейс инженера по качеству', alignment=Qt.AlignmentFlag.AlignHCenter), 0, 1)
        header_layout.setColumnStretch(0, 1)
        header_layout.setColumnStretch(1, 1)
        header_layout.setColumnStretch(2, 1)

        self.engine = sa.create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/extcaland.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)

        self.models_combobox = AdjustedComboBox()
        self.set_comboboxes()
        models_layout = QHBoxLayout()

        models_layout.addWidget(QLabel('Выберите модель: '))
        models_layout.addWidget(self.models_combobox)
        models_layout.addSpacerItem(QSpacerItem(2000, 1))

        with self.sessionmaker() as session:
            min_dt, max_dt = get_datetime_range(session)

        self.date_from = QDateTimeEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setMinimumDateTime(min_dt)
        self.date_from.setMaximumDateTime(max_dt)
        self.date_from.setDateTime(min_dt)

        self.date_to = QDateTimeEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setMinimumDateTime(min_dt)
        self.date_to.setMaximumDateTime(max_dt)
        self.date_to.setDateTime(max_dt)

        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel('С:'))
        date_layout.addWidget(self.date_from)
        date_layout.addWidget(QLabel('По:'))
        date_layout.addWidget(self.date_to)

        date_label_layout = QVBoxLayout()
        date_label_layout.addWidget(QLabel('Выбор временного диапазона'))
        date_label_layout.addLayout(date_layout)

        self.load_data_btn = QPushButton('Загрузить данные')
        self.defect_trend_btn = QPushButton('Построить тренды')

        self.load_data_btn.clicked.connect(self.load_data)
        self.defect_trend_btn.clicked.connect(self.show_defect_trend)

        data_btns_layout = QVBoxLayout()
        data_btns_layout.addWidget(self.load_data_btn)
        data_btns_layout.addWidget(self.defect_trend_btn)

        date_data_layout = QHBoxLayout()
        date_data_layout.addLayout(date_label_layout, 0)
        date_data_layout.addLayout(data_btns_layout, 1)

        self.forecast_btn = QPushButton('Спрогнозировать')

        self.result_label = QLabel()

        result_layout = QHBoxLayout()
        result_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(QLabel('Результат прогнозирования: '))
        result_layout.addWidget(self.result_label)

        frame_layout.addLayout(header_layout)
        frame_layout.addLayout(models_layout)
        frame_layout.addLayout(date_data_layout)
        frame_layout.addWidget(self.forecast_btn)
        frame_layout.addLayout(result_layout)
        frame_layout.addSpacerItem(QSpacerItem(1, 2000))

        layout.addWidget(frame)
        self.setLayout(layout)

    def set_comboboxes(self):
        with self.sessionmaker() as session:
            models = get_models(session)

        for id, name in models:
            self.models_combobox.addItem(name, id)

    def load_data(self):
        with self.sessionmaker() as session:
            model_data = get_model_data(
                session,
                self.models_combobox.currentData()
            )

            self.df = get_forecasting_data(
                session,
                model_data['parameters'],
                int(model_data['coefficients']['Window length']),
                int(model_data['coefficients']['Step']),
                self.date_from.dateTime().toPython(),
                self.date_to.dateTime().toPython(),
            )

            print(self.df)

    def show_defect_trend(self):
        if self.df is None:
            return
        
        time = self.df['timestamp'].tolist()
        values = self.df['target'].tolist()

        self.graph_window = DefectTrendWindow(time, values)
        self.graph_window.show()

    def quit_view(self):
        self.quit_view_signal.emit()
