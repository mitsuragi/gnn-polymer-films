from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QGridLayout, QHBoxLayout, QLabel, QListView, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, QFrame, QPushButton)
from PySide6.QtCore import Signal
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from gnn.inference_config import InferenceConfig
from share import AdjustedComboBox, Page, PredictionChartWidget
from gnn import DefectPredictor
from db.db_manager import get_datetime_range, get_models, get_model_data, get_training_data, get_defect_limit
from .defect_trend_window import DefectTrendWindow
from .forecast_result_window import ForecastResultWindow

class QualityEngineerView(QWidget):
    model = None
    df = None

    def __init__(self, nav):
        super().__init__()

        self.nav = nav

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
        self.forecast_btn.clicked.connect(self.forecast)

        # self.chart = PredictionChartWidget()

        result_layout = QHBoxLayout()
        result_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(QLabel('Результат прогнозирования: '))

        frame_layout.addLayout(header_layout)
        frame_layout.addLayout(models_layout)
        frame_layout.addLayout(date_data_layout)
        frame_layout.addWidget(self.forecast_btn)
        frame_layout.addLayout(result_layout)
        # frame_layout.addWidget(self.chart)
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
            self.model_data = get_model_data(
                session,
                self.models_combobox.currentData()
            )

            self.df = get_training_data(
                session,
                self.model_data['defect'],
                self.model_data['parameters'],
                self.date_from.dateTime().toPython(),
                self.date_to.dateTime().toPython()
            )

            self.limit = get_defect_limit(session, self.model_data['defect'])

    def show_defect_trend(self):
        if self.df is None:
            return
        
        df_wo_timestamp = self.df.copy()
        df_wo_timestamp.drop('timestamp', axis='columns', inplace=True)
        df_wo_timestamp['target'] = (df_wo_timestamp['target'] > self.limit).astype(int)

        self.result_window = ForecastResultWindow(self.result, df_wo_timestamp)
        self.result_window.show()

    def forecast(self):
        if self.df is None or self.model_data is None:
            return

        df_wo_timestamp = self.df.copy()
        df_wo_timestamp.drop('timestamp', axis='columns', inplace=True)
        df_wo_timestamp['target'] = (df_wo_timestamp['target'] > self.limit).astype(int)

        inf_config = InferenceConfig(
            state_dict_blob=self.model_data['model'],
            batch_size=32,
            defect_threshold=0.5,
            target_col='target',
            edge_strategy=['pearson', 'spearman'],
            edge_threshold=0.5,
            self_loops=False,
            normalize_features=True,
        )

        predictor = DefectPredictor(inf_config)
        predictor.load_model(
            model_kwargs=dict(in_channels=1, hidden_channels=64,
                              n_layers=3, heads=4, dropout=0.3)
        )

        self.result = predictor.predict(df_wo_timestamp)

        print(self.result)
        
        print(f"\nСводная таблица (первые 10 строк):")
        cols = ["prob_defect", "prediction", "true_label", "correct"]
        print(self.result.summary[cols].head(10).to_string())

    def quit_view(self):
        self.nav.navigate(Page.LOGIN)
