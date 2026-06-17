import io
from PySide6.QtWidgets import (
    QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QDateTimeEdit, QTabWidget, QSizePolicy, QSplitter, QFrame
)
from PySide6.QtCore import QThread, Qt, QDateTime, Signal
from sqlalchemy.orm import defer
from sqlalchemy.orm.session import sessionmaker
from torch.nn import Module
import torch
from pandas import DataFrame
from torch_geometric.loader import DataLoader
import numpy as np

from core.worker import Worker
from data.dataset import PolyFilmDataset
from data.splitter import make_supervised_timeseries_frame
from db.db_manager import get_model_data, get_models, get_datetime_range, get_defect_limit, get_training_data
from gnn.model import build_model
from gnn.trainer import predict

class QualityEngineerPage(QWidget):
    production_data_loaded = Signal(list, list)
    prediction_ready = Signal(list, list, list, list, float)

    _model: Module | None = None
    _df: DataFrame | None = None
    _model_data: dict | None = None
    _selected_model: int | None = None

    def __init__(self, sessionmaker: sessionmaker,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('qualityEngineerPage')
        self._sessionmaker = sessionmaker

        self._load_thread: QThread | None = None 
        self._load_worker: Worker | None = None
        self._predict_thread: QThread | None = None
        self._predict_worker: Worker | None = None

        self._build_ui()
        self._populate_models_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        title_row = QHBoxLayout()
        page_title = QLabel('Инженер по качеству')
        page_title.setObjectName('topBarTitle')
        title_row.addWidget(page_title)
        title_row.addStretch()
        root.addLayout(title_row)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setObjectName('sidebarDivider')
        root.addWidget(div)

        blocks_row = QHBoxLayout()
        blocks_row.setSpacing(14)
        blocks_row.addWidget(self._build_model_block(), stretch=5)
        blocks_row.addWidget(self._build_data_block(), stretch=3)
        blocks_row.addWidget(self._build_prediction_block(), stretch=2)
        root.addLayout(blocks_row)

        hint = QLabel("💡  Для просмотра графиков перейдите в раздел «Графики» в боковом меню.")
        hint.setObjectName("sectionLabel")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)
        root.addStretch()

    def _build_model_block(self) -> QGroupBox:
        grp = QGroupBox('Выбор модели')
        grp.setObjectName('modelSelectionGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        self.models_table = QTableWidget()
        self.models_table.setObjectName('modelsTable')
        self.models_table.setColumnCount(3)
        self.models_table.setHorizontalHeaderLabels([
            'id',
            'Название модели',
            'Прогнозируемый показатель',
        ])
        self.models_table.setColumnHidden(0, True)
        self.models_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.models_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.models_table.setSortingEnabled(True)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setMinimumHeight(130)
        layout.addWidget(self.models_table)

        self.select_model_btn = QPushButton('Выбрать модель')
        self.select_model_btn.setObjectName('primaryBtn')
        self.select_model_btn.setMinimumHeight(36)
        self.select_model_btn.clicked.connect(self.select_model)
        layout.addWidget(self.select_model_btn)

        return grp

    def _build_data_block(self) -> QGroupBox:
        grp = QGroupBox('Загрузка данных')
        grp.setObjectName('dataLoadGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        with self._sessionmaker() as session:
            min_dt, max_dt = get_datetime_range(session)

        lbl_start = QLabel('Начало периода')
        lbl_start.setObjectName('sectionLabel')
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setObjectName('startDateTime')
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setMinimumDateTime(min_dt)
        self.start_datetime.setMaximumDateTime(max_dt)
        self.start_datetime.setDateTime(min_dt)
        self.start_datetime.setMinimumHeight(34)

        lbl_end = QLabel('Конец периода')
        lbl_end.setObjectName('sectionLabel')
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setObjectName('endDateTime')
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setMinimumDateTime(min_dt)
        self.end_datetime.setMaximumDateTime(max_dt)
        self.end_datetime.setDateTime(max_dt)
        self.end_datetime.setMinimumHeight(34)

        layout.addWidget(lbl_start)
        layout.addWidget(self.start_datetime)
        layout.addWidget(lbl_end)
        layout.addWidget(self.end_datetime)
        layout.addStretch()

        self.load_data_btn = QPushButton('Загрузить данные')
        self.load_data_btn.setObjectName('primaryBtn')
        self.load_data_btn.setMinimumHeight(36)
        self.load_data_btn.clicked.connect(self.load_production_data)
        layout.addWidget(self.load_data_btn)

        return grp

    def _build_prediction_block(self) -> QGroupBox:
        grp = QGroupBox('Прогнозирование')
        grp.setObjectName('predictionGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        self.selected_model_label = QLabel('Модель не выбрана')
        self.selected_model_label.setObjectName('statusLabel')
        self.selected_model_label.setWordWrap(True)
        self.selected_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_model_label.setStyleSheet(
            "background: #F1F5F9; border: 1px solid #CBD5E1;"
            "border-radius: 4px; padding: 6px; color: #64748B;"
        )

        layout.addStretch()
        layout.addWidget(self.selected_model_label)
        layout.addStretch()

        self.run_prediction_btn = QPushButton('Выполнить прогноз')
        self.run_prediction_btn.setObjectName('primaryBtn')
        self.run_prediction_btn.setMinimumHeight(42)
        self.run_prediction_btn.clicked.connect(self.run_prediction)
        layout.addWidget(self.run_prediction_btn)

        return grp

    def _populate_models_table(self) -> None:
        with self._sessionmaker() as session:
            models = get_models(session)

        self.models_table.setRowCount(len(models))
        for row, (id, name, metric) in enumerate(models):
            self.models_table.setItem(row, 0, QTableWidgetItem(str(id)))
            self.models_table.setItem(row, 1, QTableWidgetItem(name))
            self.models_table.setItem(row, 2, QTableWidgetItem(metric))
        self.models_table.resizeRowsToContents()

    # TODO
    def select_model(self) -> None:
        selected = self.models_table.currentRow()
        if selected >= 0:
            id = self.models_table.item(selected, 0).text()
            self._selected_model = int(id) 
            self.selected_model_label.setText(id)
            self.selected_model_label.setStyleSheet(
                "background: #F0FDF4; border: 1px solid #86EFAC;"
                "border-radius: 4px; padding: 6px;"
                "color: #15803D; font-weight: 600;"
            )

            with self._sessionmaker() as session:
                self._model_data = get_model_data(session, self._selected_model)

    def load_production_data(self) -> None:
        if self._model_data is None:
            return

        if self._load_thread is not None:
            return

        from_dt = self.start_datetime.dateTime().toPython()
        to_dt = self.end_datetime.dateTime().toPython()

        self.load_data_btn.setEnabled(False)
        self.load_data_btn.setText('Загрузка...')

        self._load_thread = QThread(self)
        self._load_worker = Worker(self._load_production_data_impl, from_dt, to_dt)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_data_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_load_thread)

        self._load_thread.start()

        # with self._sessionmaker() as session:
        #     self._df = get_training_data(
        #         session,
        #         self._model_data['defect'],
        #         self._model_data['parameters'],
        #         from_dt,
        #         to_dt
        #     )
        #
        #     self._defect_limit = get_defect_limit(session, self._model_data['defect'])

        if self._df is not None:
            timestamps = self._df['timestamp'].tolist()
            values = self._df['target_raw'].tolist()

            self.production_data_loaded.emit(timestamps, values)

    def _load_production_data_impl(self, from_dt, to_dt):
        with self._sessionmaker() as session:
            df = get_training_data(
                session,
                self._model_data['defect'],
                self._model_data['parameters'],
                from_dt,
                to_dt
            )
            defect_limit = get_defect_limit(session, self._model_data['defect'])
        return df, defect_limit

    def _on_data_loaded(self, result) -> None:
        df, defect_limit = result
        self._df = df
        self._defect_limit = defect_limit

        self.load_data_btn.setEnabled(True)
        self.load_data_btn.setText('Загрузить данные')

        if df is not None:
            timestamps = df['timestamp'].tolist()
            values = df['target_raw'].tolist()
            self.production_data_loaded.emit(timestamps, values)

    def _on_load_error(self, message: str) -> None:
        self.load_data_btn.setEnabled(True)
        self.load_data_btn.setText('Загрузить данные')
        QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить данные:\n{message}')

    def _cleanup_load_thread(self) -> None:
        self._load_worker.deleteLater()
        self._load_thread.deleteLater()
        self._load_worker = None
        self._load_thread = None


    def run_prediction(self) -> None:
        if self._df is None or self._model_data is None:
            return

        if self._predict_thread is not None:
            return

        self.run_prediction_btn.setEnabled(False)
        self.run_prediction_btn.setText('Выполняется...')

        self._predict_thread = QThread(self)
        self._predict_worker = Worker(self._run_prediction_impl)
        self._predict_worker.moveToThread(self._predict_thread)

        self._predict_thread.started.connect(self._predict_worker.run)
        self._predict_worker.finished.connect(self._on_prediction_ready)
        self._predict_worker.error.connect(self._on_prediction_error)
        self._predict_worker.finished.connect(self._predict_thread.quit)
        self._predict_worker.error.connect(self._predict_thread.quit)
        self._predict_thread.finished.connect(self._cleanup_predict_thread)

        self._predict_thread.start()

    def _run_prediction_impl(self):
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        buffer = io.BytesIO(self._model_data['model'])
        bundle = torch.load(buffer, map_location=device, weights_only=False)

        df = make_supervised_timeseries_frame(
            self._df,
            target_col='target_raw',
            time_col='timestamp',
            target_threshold=(self._defect_limit - 30),
            forecast_horizon=10,
            lags=(1, 5, 20),
            rolling_windows=(20,),
            add_current=True,
        )

        feature_cols = [c for c in df.columns if c not in ['timestamp', 'target']]

        ds = PolyFilmDataset(
            df[feature_cols + ['target']],
            target_col='target',
            normalize_features=True,
            feature_mean=bundle['feature_mean'],
            feature_std=bundle['feature_std'],
            edge_index=bundle['edge_index'],
            edge_attr=bundle['edge_attr'],
        )
        
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        model = build_model(
            n_nodes=len(bundle['feature_names']),
            node_emb_dim=16,
            in_channels=1,
            hidden_channels=32,
            n_layers=2,
            heads=2,
            dropout=0.5,
            edge_dim=1,
            pooling='concat'
        )

        model.load_state_dict(bundle['model_state_dict'])
        model.to(device)
        model.eval()

        true, prob, pred = predict(model, loader, bundle['best_threshold'])
        tp = int(np.sum((pred == 1) & (true == 1)))
        tn = int(np.sum((pred == 0) & (true == 0)))
        fp = int(np.sum((pred == 1) & (true == 0)))
        fn = int(np.sum((pred == 0) & (true == 1)))
        answer_distribution = [tp, tn, fp, fn]

        timestamps = df['timestamp'].tolist()
        
        return timestamps, true, prob, answer_distribution, bundle['best_threshold']

    def _on_prediction_ready(self, result) -> None:
        timestamps, true, pred, prob, threshold = result

        self.run_prediction_btn.setEnabled(True)
        self.run_prediction_btn.setText('Выполнить прогноз')

        self.prediction_ready.emit(timestamps, true, pred, prob, threshold)

    def _on_prediction_error(self, message: str) -> None:
        self.run_prediction_btn.setEnabled(True)
        self.run_prediction_btn.setText('Выполнить прогноз')
        QMessageBox.critical(self, 'Ошибка', f'Не удалось выполнить прогноз:\n{message}')

    def _cleanup_predict_thread(self) -> None:
        self._predict_worker.deleteLater()
        self._predict_thread.deleteLater()
        self._predict_worker = None
        self._predict_thread = None
