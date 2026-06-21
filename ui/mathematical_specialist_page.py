import io

from PySide6.QtWidgets import (
    QFrame, QListView, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QDateTimeEdit, QSplitter,
    QSpinBox, QTextEdit, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QThread
from sqlalchemy.orm.session import sessionmaker
import torch
from torch.nn import Module
from pandas import DataFrame
from torch_geometric.loader import DataLoader
 
from core.worker import Worker
from data.dataset import PolyFilmDataset
from data.splitter import make_supervised_timeseries_frame, temporal_episode_split
from gnn.metrics_logger import MetricsLogger
from gnn.train_config import TrainConfig
from share.AdjustedComboBox import AdjustedComboBox
from share.ParametersListModel import ParametersListModel
from db.db_manager import delete_model, get_defect_limit, get_defects, get_models, get_datetime_range, get_nn_coeffs, get_parameters, get_training_data, save_model
from gnn.trainer import train
from gnn.model import build_model

class MathematicalSpecialistPage(QWidget):
    training_data_loaded = Signal(list, list)
    training_complete = Signal(MetricsLogger)

    _model: Module | None = None
    _df: DataFrame | None = None
    _best_threshold: float | None = None
    _logger: MetricsLogger | None = None
    _metrics = {
        'Precision': 0.0,
        'Recall': 0.0,
        'F1-score': 0.0,
        'PR-AUC': 0.0,
    }

    def __init__(self, sessionmaker: sessionmaker,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('mathSpecialstPage')
        self._sessionmaker = sessionmaker

        self._load_thread: QThread | None = None
        self._load_worker: Worker | None = None
        self._train_thread: QThread | None = None
        self._train_worker: Worker | None = None

        self._build_ui()
        self._populate_saved_models()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        title_row = QHBoxLayout()
        page_title = QLabel('Специалист по математическому обеспечению')
        page_title.setObjectName('topBarTitle')
        title_row.addWidget(page_title)
        title_row.addStretch()
        root.addLayout(title_row)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setObjectName('sidebarDivider')
        root.addWidget(div)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName('mathSplitter')

        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(12)
        row1_layout.addWidget(self._build_data_block(), stretch=2)
        row1_layout.addWidget(self._build_model_params_block(), stretch=2)
        row1_layout.addWidget(self._build_film_params_block(), stretch=4)
        splitter.addWidget(row1)

        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(12)
        row2_layout.addWidget(self._build_training_block(), stretch=5)
        row2_layout.addWidget(self._build_metrics_block(), stretch=1)
        splitter.addWidget(row2)

        row3 = QWidget()
        row3_layout = QHBoxLayout(row3)
        row3_layout.setContentsMargins(0, 0, 0, 0)
        row3_layout.setSpacing(12)
        row3_layout.addWidget(self._build_save_block(), stretch=2)
        row3_layout.addWidget(self._build_manage_block(), stretch=5)
        splitter.addWidget(row3)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 4)

        root.addWidget(splitter, 1)

        hint = QLabel("💡  Для просмотра графиков обучения перейдите в раздел «Графики» в боковом меню.")
        hint.setObjectName("sectionLabel")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)

    def _build_data_block(self) -> QGroupBox:
        grp = QGroupBox('Данные для обучения')
        grp.setObjectName('trainingDataGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        with self._sessionmaker() as session:
            min_dt, max_dt = get_datetime_range(session)

        lbl_start = QLabel("Начало периода")
        lbl_start.setObjectName("sectionLabel")
        self.train_start = QDateTimeEdit()
        self.train_start.setObjectName("trainStartDatetime")
        self.train_start.setCalendarPopup(True)
        self.train_start.setMinimumDateTime(min_dt)
        self.train_start.setMaximumDateTime(max_dt)
        self.train_start.setDateTime(min_dt)
        self.train_start.setMinimumHeight(34)
 
        lbl_end = QLabel("Конец периода")
        lbl_end.setObjectName("sectionLabel")
        self.train_end = QDateTimeEdit()
        self.train_end.setObjectName("trainEndDatetime")
        self.train_end.setCalendarPopup(True)
        self.train_end.setMinimumDateTime(min_dt)
        self.train_end.setMaximumDateTime(max_dt)
        self.train_end.setDateTime(max_dt)
        self.train_end.setMinimumHeight(34)

        layout.addWidget(lbl_start)
        layout.addWidget(self.train_start)
        layout.addWidget(lbl_end)
        layout.addWidget(self.train_end)
        layout.addStretch()

        self.load_training_btn = QPushButton('Загрузить данные')
        self.load_training_btn.setObjectName('primaryBtn')
        self.load_training_btn.setMinimumHeight(38)
        self.load_training_btn.clicked.connect(self.load_training_data)
        layout.addWidget(self.load_training_btn)

        return grp

    def _build_model_params_block(self) -> QGroupBox:
        grp = QGroupBox('Параметры модели')
        grp.setObjectName('modelParamsGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        lbl_epochs = QLabel('Макс. эпох обучения')
        lbl_epochs.setObjectName('sectionLabel')
        self.epochs_combobox = AdjustedComboBox()
        self.epochs_combobox.setObjectName("epochsComboBox")
        self.epochs_combobox.setMinimumHeight(34)

        lbl_batch = QLabel("Размер батча")
        lbl_batch.setObjectName("sectionLabel")
        self.batch_combobox = AdjustedComboBox()
        self.batch_combobox.setObjectName("batchComboBox")
        self.batch_combobox.setMinimumHeight(34)
 
        lbl_layers = QLabel("Количество слоёв")
        lbl_layers.setObjectName("sectionLabel")
        self.layers_spinbox = QSpinBox()
        self.layers_spinbox.setObjectName("layersSpinBox")
        self.layers_spinbox.setRange(1, 8)
        self.layers_spinbox.setValue(2)
        self.layers_spinbox.setSingleStep(1)
        self.layers_spinbox.setMinimumHeight(34)

        self._fill_model_params()

        layout.addWidget(lbl_epochs)
        layout.addWidget(self.epochs_combobox)
        layout.addWidget(lbl_batch)
        layout.addWidget(self.batch_combobox)
        layout.addWidget(lbl_layers)
        layout.addWidget(self.layers_spinbox)
        layout.addStretch()

        return grp

    def _build_film_params_block(self) -> QGroupBox:
        grp = QGroupBox('Параметры производства')
        grp.setObjectName('filmParamsGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        lbl_defect = QLabel('Показатель качества')
        lbl_defect.setObjectName('sectionaLabel')
        self.defect_combobox = AdjustedComboBox()

        lbl_params = QLabel('Управляющие воздействия')
        lbl_params.setObjectName('sectionLabel')
        self.parameters_view = QListView()
        self.parameters_view.setObjectName('parametersView')
        self.parameters_model = ParametersListModel([])
        self.parameters_model.setObjectName('parametersModel')
        self.parameters_view.setUniformItemSizes(True)
        self.parameters_view.setSelectionMode(QListView.SelectionMode.NoSelection)
        self.parameters_view.setModel(self.parameters_model)

        self._fill_film_params()

        layout.addWidget(lbl_defect)
        layout.addWidget(self.defect_combobox)
        layout.addWidget(lbl_params)
        layout.addWidget(self.parameters_view)
        layout.addStretch()

        return grp

    def _build_training_block(self) -> QGroupBox:
        grp = QGroupBox('Обучение модели')
        grp.setObjectName('trainingGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        self.train_btn = QPushButton('Обучить модель')
        self.train_btn.setObjectName('primaryBtn')
        self.train_btn.setMinimumHeight(38)
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)

        lbl_log = QLabel("Журнал обучения")
        lbl_log.setObjectName("sectionLabel")
        self.train_log = QTextEdit()
        self.train_log.setObjectName("trainLog")
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText(
            "Лог обучения будет отображаться здесь...\n"
            "Epoch 1/100 — loss: 0.8934 — val_loss: 0.9102\n"
            "Epoch 2/100 — loss: 0.7521 — val_loss: 0.7830\n"
            "..."
        )
        self.train_log.setMinimumHeight(80)

        layout.addWidget(lbl_log)
        layout.addWidget(self.train_log)

        return grp

    def _build_metrics_block(self) -> QGroupBox:
        grp = QGroupBox('Метрики модели')
        grp.setObjectName('metricsModelGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        self.metrics_table = QTableWidget()
        self.metrics_table.setObjectName('metricsModelTable')
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels([
            'Метрика', 'Значение'
        ])
        self.metrics_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection)
        self.metrics_table.setSortingEnabled(False)
        self.metrics_table.setAlternatingRowColors(False)
        self.metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)

        self._set_metrics_table()

        layout.addWidget(self.metrics_table)

        return grp

    def _build_save_block(self) -> QGroupBox:
        grp = QGroupBox('Сохранение модели')
        grp.setObjectName('saveModelGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        lbl = QLabel('Название модели')
        lbl.setObjectName('sectionLabel')
        self.model_name_input = QLineEdit()
        self.model_name_input.setObjectName('modelNameInput')
        self.model_name_input.setMinimumHeight(36)

        self.save_model_btn = QPushButton('Сохранить модель')
        self.save_model_btn.setObjectName('primaryBtn')
        self.save_model_btn.setMinimumHeight(38)
        self.save_model_btn.clicked.connect(self.save_model)

        layout.addWidget(lbl)
        layout.addWidget(self.model_name_input)
        layout.addStretch()
        layout.addWidget(self.save_model_btn)

        return grp

    def _build_manage_block(self) -> QGroupBox:
        grp = QGroupBox('Сохраненные модели')
        grp.setObjectName('savedModelsGroup')
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        self.saved_models_table = QTableWidget()
        self.saved_models_table.setObjectName("savedModelsTable")
        self.saved_models_table.setColumnCount(3)
        self.saved_models_table.setHorizontalHeaderLabels([
            "id", "Название", "Прогнозируемый показатель",
        ])
        self.saved_models_table.setColumnHidden(0, True)
        self.saved_models_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.saved_models_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        self.saved_models_table.setSortingEnabled(True)
        self.saved_models_table.setAlternatingRowColors(True)
        self.saved_models_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.saved_models_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.saved_models_table.verticalHeader().setVisible(False)
        layout.addWidget(self.saved_models_table)

        self.delete_model_btn = QPushButton('Удалить модель')
        self.delete_model_btn.setObjectName('dangerButton')
        self.delete_model_btn.setMinimumHeight(36)
        self.delete_model_btn.clicked.connect(self.delete_model)
        layout.addWidget(self.delete_model_btn)

        return grp

    def _populate_saved_models(self) -> None:
        with self._sessionmaker() as session:
            models = get_models(session)

        self.saved_models_table.setRowCount(len(models))
        for row, (id, name, metric) in enumerate(models):
            self.saved_models_table.setItem(row, 0, QTableWidgetItem(str(id)))
            self.saved_models_table.setItem(row, 1, QTableWidgetItem(name))
            self.saved_models_table.setItem(row, 2, QTableWidgetItem(metric))
        self.saved_models_table.resizeRowsToContents()

    def _fill_model_params(self) -> None:
        with self._sessionmaker() as session:
            nn_coefficients_data = get_nn_coeffs(session)

            for coef in nn_coefficients_data:
                if coef.IdCoefficientType == 1:
                    self.batch_combobox.addItem(coef.Value, coef.IdCoefficient)
                elif coef.IdCoefficientType == 3:
                    self.epochs_combobox.addItem(coef.Value, coef.IdCoefficient)

    def _fill_film_params(self) -> None:
        with self._sessionmaker() as session:
            defects_data = get_defects(session)
            for name, id in defects_data:
                self.defect_combobox.addItem(name, id)

            parameters_data = get_parameters(session)
            self.parameters_model.setItemsList(parameters_data)

    def _set_metrics_table(self) -> None:
        self.metrics_table.setRowCount(len(self._metrics))
        for row, (metric, value) in enumerate(self._metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f'{value:.4f}'))

    def load_training_data(self) -> None:
        if self._load_thread is not None:
            return

        defect_id = self.defect_combobox.currentData()

        parameters = self.parameters_model.getCheckedIds()
        from_dt = self.train_start.dateTime().toPython()
        to_dt = self.train_end.dateTime().toPython()

        self.load_training_btn.setEnabled(False)
        self.load_training_btn.setText('Загрузка...')

        self._load_thread = QThread(self)
        self._load_worker = Worker(self._load_training_data_impl, defect_id, parameters, from_dt, to_dt)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_data_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_load_thread)

        self._load_thread.start()

    def _load_training_data_impl(self, defect_id, parameters, from_dt, to_dt):
        with self._sessionmaker() as session:
            df = get_training_data(
                session,
                defect_id,
                parameters,
                from_dt,
                to_dt
            )

            defect_limit = get_defect_limit(session, defect_id)

        return df, defect_limit

    def _on_data_loaded(self, result) -> None:
        df, defect_limit = result
        self._df, self._defect_limit = df, defect_limit

        self.load_training_btn.setEnabled(True)
        self.load_training_btn.setText('Загрузить данные')

        if self._df is not None:
            timestamps = self._df['timestamp'].tolist()
            values = self._df['target_raw'].tolist()

            self.training_data_loaded.emit(timestamps, values)

    def _on_load_error(self, message: str) -> None:
        self.load_training_btn.setEnabled(True)
        self.load_training_btn.setText('Загрузить данные')
        QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить данные:\n{message}')

    def _cleanup_load_thread(self) -> None:
        self._load_worker.deleteLater()
        self._load_thread.deleteLater()
        self._load_worker = None
        self._load_thread = None

    def train_model(self) -> None:
        if self._df is None:
            return

        if self._train_thread is not None:
            return

        self.train_btn.setEnabled(False)
        self.train_btn.setText('Выполняется')

        self._train_thread = QThread(self)
        self._train_worker = Worker(self._train_model_impl)
        self._train_worker.moveToThread(self._train_thread)
        
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        self._train_thread.finished.connect(self._cleanup_train_thread)

        self._train_thread.start()

    def _train_model_impl(self):
        FORECAST_HORIZON = 10

        df = make_supervised_timeseries_frame(
            self._df,
            target_col='target_raw',
            time_col='timestamp',
            target_threshold=(self._defect_limit - 30),
            forecast_horizon=FORECAST_HORIZON,
            lags=(1, 5, 20),
            rolling_windows=(20,),
            add_current=True,
        )

        splits = temporal_episode_split(
            df,
            target_col='target',
            time_col='timestamp',
            train_episode_size=0.65,
            val_episode_size=0.1,
            test_episode_size=0.25,
            event_max_gap=50,
            purge_gap=70,
        )

        feature_cols = [c for c in df.columns if c not in ['timestamp', 'target']]

        train_ds = PolyFilmDataset(
            splits.train[feature_cols + ['target']],
            target_col='target',
            edge_strategy=['pearson', 'spearman'],
            threshold=0.7,
            top_k_edges_per_node=128,
        )

        self._feature_mean = train_ds.feature_mean
        self._feature_std = train_ds.feature_std
        self._edge_index = train_ds.edge_index
        self._edge_attr = train_ds.edge_attr
        self._feature_names = train_ds.feature_names

        def make_eval_dataset(split_df):
            return PolyFilmDataset(
                split_df[feature_cols + ['target']],
                target_col='target',
                normalize_features=True,
                feature_mean=train_ds.feature_mean,
                feature_std=train_ds.feature_std,
                edge_index=train_ds.edge_index,
                edge_attr=train_ds.edge_attr,
            )

        val_ds = make_eval_dataset(splits.val)
        test_ds = make_eval_dataset(splits.test)

        batch_size = int(self.batch_combobox.currentText())
        n_layers = self.layers_spinbox.value()
        n_epochs = int(self.epochs_combobox.currentText())

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = build_model(
            n_nodes=len(train_ds.feature_names),
            node_emb_dim=16,
            in_channels=train_ds.num_node_features,
            hidden_channels=32,
            n_layers=n_layers,
            heads=2,
            dropout=0.5,
            edge_dim=1,
            pooling='concat'
        )

        config = TrainConfig(
            n_epochs=n_epochs,
            learning_rate=5e-4,
            use_sampler=True,
            sampler_kind='event_aware',
            sampler_pos_ratio=0.30,
            sampler_context_radius=70,
            sampler_hard_neg_ratio=0.60,
            defect_threshold=0.5,
            threshold_metric='f1',
            threshold_min=0.01,
            threshold_max=0.90,
            weight_decay=3e-4,
            batch_size=batch_size,
            es_patience=5,
            es_monitor='pr_auc',
            es_mode='max',
            checkpoint_path='best_model.pt',
            log_csv_path='training_log.csv',
        )

        train_labels = splits.train['target'].values

        logger, model, best_threshold = train(model, train_ds, val_loader, test_loader, config, train_labels=train_labels)

        return logger, model, best_threshold

    def _on_train_finished(self, result) -> None:
        self._logger, self._model, self._best_threshold = result

        self.train_btn.setEnabled(True)
        self.train_btn.setText('Обучить модель')

        test_metrics = self._logger.history[-1]
        self._metrics['Precision'] = test_metrics.precision
        self._metrics['Recall'] = test_metrics.recall
        self._metrics['F1-score'] = test_metrics.f1
        self._metrics['PR-AUC'] = test_metrics.pr_auc
        self._set_metrics_table()

        self.training_complete.emit(self._logger)

    def _on_train_error(self, message: str) -> None:
        self.train_btn.setEnabled(True)
        self.train_btn.setText('Обучить модель')
        
        QMessageBox.critical(self, 'Ошибка', f'При обучении возникла ошибка:\n{message}')

    def _cleanup_train_thread(self) -> None:
        self._train_worker.deleteLater()
        self._train_thread.deleteLater()
        self._train_worker = None
        self._train_thread = None

    def save_model(self) -> None:
        if self._model:
            buffer = io.BytesIO()
            bundle = {
                'model_state_dict': self._model.state_dict(),
                'best_threshold': self._best_threshold,
                'feature_mean': self._feature_mean,
                'feature_std': self._feature_std,
                'edge_index': self._edge_index,
                'edge_attr': self._edge_attr,
                'feature_names': self._feature_names,
            }
            torch.save(bundle, buffer)

            model_blob = buffer.getvalue()

            coefficients = []

            coefficients.append(self.batch_combobox.currentData())
            coefficients.append(self.epochs_combobox.currentData())
            # coefficients.append(self.layers_spinbox.value())

            model_name = self.model_name_input.text()
            from_dt = self.train_start.dateTime().toPython()
            to_dt = self.train_end.dateTime().toPython()
            parameters = self.parameters_model.getCheckedIds()
            defect = self.defect_combobox.currentData()

            with self._sessionmaker() as session:
                save_model(
                    session,
                    model_name,
                    from_dt,
                    to_dt,
                    model_blob,
                    self._best_threshold,
                    parameters,
                    defect,
                    coefficients
                )

            self._populate_saved_models()

    def delete_model(self) -> None:
        selected = self.saved_models_table.currentRow()
        if selected >= 0:
            id = int(self.saved_models_table.item(selected, 0).text())

            with self._sessionmaker() as session:
                delete_model(session, id)

        self._populate_saved_models()

    def append_log(self, message: str) -> None:
        self.train_log.append(message)
