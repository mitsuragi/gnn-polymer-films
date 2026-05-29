from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QDateTimeEdit, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListView, QSizePolicy, QSpacerItem, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QFrame, QPushButton, QLineEdit)
from sqlalchemy.orm import sessionmaker
import sqlalchemy as sa
import torch
from torch_focalloss import BinaryFocalLoss
from torch_geometric.loader import DataLoader
import io

from gnn import build_model, TrainConfig, train
from share import ParametersListModel, AdjustedComboBox, Page
from db.db_manager import get_parameters, get_defects, get_datetime_range, get_nn_coeffs, get_training_data, save_model, delete_model, get_models, get_defect_limit
from .defect_trend_window import DefectTrendWindow
from gnn import PolymerGAT
from data import PolyFilmDataset, stratified_split

class MathSpecialistView(QWidget):
    model = None
    df = None

    def __init__(self, nav):
        super().__init__()

        self.nav = nav

        self.setWindowTitle('Интерфейс специалиста по математическому обеспечению')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(15, 10, 15, 10)
        frame_layout.setSpacing(5)

        self.quit_view_btn = QPushButton('Выйти')
        self.quit_view_btn.clicked.connect(self.quit_view)

        header_layout = QGridLayout()
        header_layout.addWidget(self.quit_view_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(QLabel('Интерфейс специалиста по математическому обеспечению', alignment=Qt.AlignmentFlag.AlignHCenter), 0, 1)
        header_layout.setColumnStretch(0, 1)
        header_layout.setColumnStretch(1, 1)
        header_layout.setColumnStretch(2, 1)

        self.train_label = QLabel('Обучение модели')
        self.train_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.model_name_textbox = QLineEdit(placeholderText='Введите название модели')
        self.model_name_textbox.setFixedWidth(250)
        
        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel('Название модели: '), 0)
        model_name_layout.addWidget(self.model_name_textbox, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft)

        settings_combobox_layout = QHBoxLayout()
        settings_combobox_layout.setSpacing(6) 

        self.layers_combobox = AdjustedComboBox()
        self.batch_combobox = AdjustedComboBox()
        self.epoch_combobox = AdjustedComboBox()
        self.step_combobox = AdjustedComboBox()
        self.window_combobox = AdjustedComboBox()

        # settings_combobox_layout.addWidget(QLabel('Количество скрытых слоев'))
        # settings_combobox_layout.addWidget(self.layers_combobox)
        settings_combobox_layout.addWidget(QLabel('Размер батча'))
        settings_combobox_layout.addWidget(self.batch_combobox)
        settings_combobox_layout.addWidget(QLabel('Максимальное количество эпох'))
        settings_combobox_layout.addWidget(self.epoch_combobox)
        settings_combobox_layout.addWidget(QLabel('Шаг измерений'))
        settings_combobox_layout.addWidget(self.step_combobox)
        settings_combobox_layout.addWidget(QLabel('Длина окна'))
        settings_combobox_layout.addWidget(self.window_combobox)
        settings_combobox_layout.addSpacerItem(QSpacerItem(1000, 1))

        settings_layout = QVBoxLayout()
        settings_layout.addLayout(model_name_layout)
        settings_layout.addLayout(settings_combobox_layout)

        self.engine = sa.create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/extcaland.db', echo=False)
        self.sessionmaker = sessionmaker(bind=self.engine)
        with self.sessionmaker() as session:
            parameters_data = get_parameters(session)

        forecasting_layout = QGridLayout()

        self.forecasting_label = QLabel('Настройки прогнозирования')

        self.defects_label = QLabel('Прогнозируемый показатель качества')
        self.defects_combobox = AdjustedComboBox()
        self.defects_combobox.setSizePolicy(QSizePolicy.Policy.Maximum,
                                            QSizePolicy.Policy.Preferred)
        self.set_comboboxes()

        self.parameters_label = QLabel('Управляющие воздействия')
        self.parameters_view = QListView()
        self.parameters_model = ParametersListModel(parameters_data)

        self.parameters_view.setUniformItemSizes(True)
        self.parameters_view.setSelectionMode(QListView.NoSelection)
        self.parameters_view.setModel(self.parameters_model)

        forecasting_layout.addWidget(self.forecasting_label, 0, 0)
        forecasting_layout.addWidget(self.defects_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        forecasting_layout.addWidget(self.defects_combobox, 1, 1)
        forecasting_layout.addWidget(self.parameters_label, 2, 0, Qt.AlignmentFlag.AlignRight)
        forecasting_layout.addWidget(self.parameters_view, 2, 1)

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

        self.train_btn = QPushButton('Обучение')
        self.train_btn.clicked.connect(self.training)

        verification_layout = QGridLayout()
        verification_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.mae_label = QLabel()
        self.rmse_label = QLabel()
        self.wape_label = QLabel()
        self.precision_label = QLabel()
        self.recall_label = QLabel()

        verification_layout.addWidget(QLabel('MAE = '), 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        verification_layout.addWidget(self.mae_label, 0, 1)
        verification_layout.addWidget(QLabel('RMSE = '), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
        verification_layout.addWidget(self.rmse_label, 1, 1)
        verification_layout.addWidget(QLabel('WAPE = '), 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        verification_layout.addWidget(self.wape_label, 2, 1)
        verification_layout.addWidget(QLabel('Precision = '), 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        verification_layout.addWidget(self.precision_label, 0, 3)
        verification_layout.addWidget(QLabel('Recall = '), 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        verification_layout.addWidget(self.recall_label, 1, 3)

        verification_layout.setColumnStretch(0, 0)
        verification_layout.setColumnStretch(1, 1)
        verification_layout.setColumnStretch(2, 0)
        verification_layout.setColumnStretch(3, 1)

        plot_layout = QHBoxLayout()
        self.compare_plot_btn = QPushButton('График реальных и спрогнозированных величин')
        self.training_plot_btn = QPushButton('График обучения модели')
        plot_layout.addWidget(self.compare_plot_btn)
        plot_layout.addWidget(self.training_plot_btn)

        self.save_model_btn = QPushButton('Сохранить модель')
        self.save_model_btn.clicked.connect(self.save_model)

        self.table_models = QTableWidget()
        self.table_models.setColumnCount(1)
        self.table_models.setHorizontalHeaderLabels(['Название'])
        self.table_models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_models.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        header = self.table_models.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.update_table()

        self.delete_model_btn = QPushButton('Удалить модель')
        self.delete_model_btn.clicked.connect(self.delete_model)

        models_layout = QVBoxLayout()
        models_layout.addWidget(self.table_models)
        models_layout.addWidget(self.delete_model_btn)

        frame_layout.addLayout(header_layout)
        frame_layout.addWidget(QLabel('Обучение модели'))
        frame_layout.addLayout(settings_layout)
        frame_layout.addLayout(forecasting_layout)
        frame_layout.addLayout(date_data_layout)
        frame_layout.addWidget(self.train_btn)
        frame_layout.addWidget(QLabel('Верификация модели'))
        frame_layout.addLayout(verification_layout)
        frame_layout.addLayout(plot_layout)
        frame_layout.addWidget(self.save_model_btn)
        frame_layout.addLayout(models_layout)

        layout.addWidget(frame)
        self.setLayout(layout)

    def set_comboboxes(self):
        with self.sessionmaker() as session:
            defects_data = get_defects(session)
            for name, id in defects_data:
                self.defects_combobox.addItem(name, id)

            nn_coefficient_data = get_nn_coeffs(session)

            for coef in nn_coefficient_data:
                if coef.IdCoefficientType == 1:
                    self.batch_combobox.addItem(coef.Value, coef.IdCoefficient)
                elif coef.IdCoefficientType == 3:
                    self.epoch_combobox.addItem(coef.Value, coef.IdCoefficient)
                elif coef.IdCoefficientType == 4:
                    self.window_combobox.addItem(coef.Value, coef.IdCoefficient)
                elif coef.IdCoefficientType == 5:
                    self.step_combobox.addItem(coef.Value, coef.IdCoefficient)

    def training(self):
        if self.df is None:
            return

        df_wo_timestamp = self.df.copy()
        df_wo_timestamp.drop('timestamp', axis='columns', inplace=True)
        df_wo_timestamp['target'] = (df_wo_timestamp['target'] > self.limit).astype(int)

        splits = stratified_split(df_wo_timestamp, target_col='target')
        
        def make_dataset(split_df):
            return PolyFilmDataset(
                split_df,
                target_col='target',
                edge_strategy=['pearson', 'spearman'],
                threshold=0.5,
                self_loops=False,
            )

        train_ds = make_dataset(splits.train)
        val_ds = make_dataset(splits.val)
        test_ds = make_dataset(splits.test)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        test_loader = DataLoader(test_ds, batch_size=64)

        self.model = build_model(
            in_channels=train_ds.num_node_features,
            hidden_channels=64,
            n_layers=3,
            heads=4,
            dropout=0.3,
            edge_dim=1,
            pooling='concat',
        )

        config = TrainConfig(
            n_epochs        = 100,
            learning_rate   = 1e-4,
            weight_decay    = 1e-4,
            es_patience     = 10,
            es_monitor      = "loss",
            es_mode         = "min",
            focal_alpha     = 0.95,
            focal_gamma     = 2.0,
            checkpoint_path = "best_model.pt",
            log_csv_path    = "training_log.csv",
        )

        logger = train(self.model, train_loader, val_loader, test_loader, config)
        best = logger.best(phase='test', key='loss', mode='min')
        self.set_metrics(best.mae, best.rmse, best.wape, best.precision, best.recall)

    def save_model(self):
        if self.model:
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)

            model_blob = buffer.getvalue()

            coefficients = []

            coefficients.append(self.batch_combobox.currentData())
            coefficients.append(self.epoch_combobox.currentData())
            coefficients.append(self.step_combobox.currentData())
            coefficients.append(self.window_combobox.currentData())

            with self.sessionmaker() as session:
                save_model(
                    session,
                    self.model_name_textbox.text(),
                    self.date_from.dateTime().toPython(),
                    self.date_to.dateTime().toPython(),
                    model_blob,
                    self.parameters_model.getCheckedIds(),
                    self.defects_combobox.currentData(),
                    coefficients
                )

            self.update_table()

    def delete_model(self):
        row = self.table_models.currentRow()

        model = self.table_models.item(row, 0)

        if model is not None:
            model_id = model.data(1)

            with self.sessionmaker() as session:
                delete_model(session, model_id)

        self.update_table()

    def set_metrics(self, mae, rmse, wape, prec, rec):
        self.mae_label.setText(f"{mae:.3f}")
        self.rmse_label.setText(f"{rmse:.3f}")
        self.wape_label.setText(f"{wape:.3f}")
        self.precision_label.setText(f"{prec:.3f}")
        self.recall_label.setText(f"{rec:.3f}")

    def update_table(self):
        with self.sessionmaker() as session:
            models = get_models(session)

        self.table_models.setRowCount(len(models))

        for row, (id, name) in enumerate(models):
            item = QTableWidgetItem(name)

            item.setData(1, id)
               
            self.table_models.setItem(row, 0, item)

    def show_defect_trend(self):
        if self.df is None:
            return
        
        time = self.df['timestamp'].tolist()
        values = self.df['target'].tolist()

        self.graph_window = DefectTrendWindow(time, values)
        self.graph_window.show()

    def load_data(self):
        defect_id = self.defects_combobox.currentData(),
        with self.sessionmaker() as session:
            self.df = get_training_data(
                session,
                self.defects_combobox.currentData(),
                self.parameters_model.getCheckedIds(),
                self.date_from.dateTime().toPython(),
                self.date_to.dateTime().toPython()
            )

            self.limit = get_defect_limit(session, self.defects_combobox.currentData())

    def quit_view(self):
        self.nav.navigate(Page.LOGIN)
