from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QGridLayout, QHBoxLayout, QLabel, QListView, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, QFrame, QPushButton)
from sqlalchemy.orm import sessionmaker

from share import ParametersListModel, AdjustedComboBox
from db.db_manager import get_parameters, get_defects, create_engine, get_datetime_range, get_nn_coeffs, get_training_data
from data.dataset import PolyFilmDataset

class MathSpecialistView(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Интерфейс специалиста по математическому обеспечению')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(15, 10, 15, 10)
        frame_layout.setSpacing(20)

        self.label = QLabel('Интерфейс специалиста по математическому обеспечению')
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.train_label = QLabel('Обучение модели')
        self.train_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(6) 

        self.layers_combobox = AdjustedComboBox()
        self.batch_combobox = AdjustedComboBox()
        self.epoch_combobox = AdjustedComboBox()
        self.step_combobox = AdjustedComboBox()
        self.window_combobox = AdjustedComboBox()

        settings_layout.addWidget(QLabel('Количество скрытых слоев'))
        settings_layout.addWidget(self.layers_combobox)
        settings_layout.addWidget(QLabel('Размер батча'))
        settings_layout.addWidget(self.batch_combobox)
        settings_layout.addWidget(QLabel('Максимальное количество эпох'))
        settings_layout.addWidget(self.epoch_combobox)
        settings_layout.addWidget(QLabel('Шаг измерений'))
        settings_layout.addWidget(self.step_combobox)
        settings_layout.addWidget(QLabel('Длина окна'))
        settings_layout.addWidget(self.window_combobox)
        settings_layout.addSpacerItem(QSpacerItem(1000, 1))

        self.engine = create_engine('sqlite:////home/mitsuri/Code/Python/gnn/extcaland/extcaland.db')
        self.sessionmaker = sessionmaker(bind=self.engine)
        with self.sessionmaker() as session:
            parameters_data = get_parameters(session)
            defects_data = get_defects(session)
        print(defects_data)

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
        date_layout.addSpacerItem(QSpacerItem(2000,1))

        self.train_btn = QPushButton('Обучение')
        self.train_btn.clicked.connect(self.training)

        verification_layout = QGridLayout()

        self.mae_label = QLabel()
        self.rmse_label = QLabel()
        self.wape_label = QLabel()
        self.precision_label = QLabel()
        self.recall_label = QLabel()

        verification_layout.addWidget(QLabel('MAE = '), 0, 0)
        verification_layout.addWidget(self.mae_label, 0, 1)
        verification_layout.addWidget(QLabel('RMSE = '), 1, 0)
        verification_layout.addWidget(self.rmse_label, 1, 1)
        verification_layout.addWidget(QLabel('WAPE = '), 2, 0)
        verification_layout.addWidget(self.wape_label, 2, 1)
        verification_layout.addWidget(QLabel('Precision = '), 0, 2)
        verification_layout.addWidget(self.precision_label, 0, 3)
        verification_layout.addWidget(QLabel('Recall = '), 1, 2)
        verification_layout.addWidget(self.recall_label, 1, 3)

        plot_layout = QHBoxLayout()
        self.compare_plot_btn = QPushButton('График реальных и спрогнозированных величин')
        self.training_plot_btn = QPushButton('График обучения модели')
        plot_layout.addWidget(self.compare_plot_btn)
        plot_layout.addWidget(self.training_plot_btn)

        self.save_model_btn = QPushButton('Сохранить модель')
        self.save_model_btn.clicked.connect(self.save_model)

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(QLabel('Обучение модели'))
        frame_layout.addLayout(settings_layout)
        frame_layout.addLayout(forecasting_layout)
        frame_layout.addWidget(QLabel('Выбор временного диапазона'))
        frame_layout.addLayout(date_layout)
        frame_layout.addWidget(self.train_btn)
        frame_layout.addWidget(QLabel('Верификация модели'))
        frame_layout.addLayout(verification_layout)
        frame_layout.addLayout(plot_layout)
        frame_layout.addWidget(self.save_model_btn)
        frame_layout.addSpacerItem(QSpacerItem(1, 1000))

        layout.addWidget(frame)
        self.setLayout(layout)

    def set_comboboxes(self):
        with self.sessionmaker() as session:
            defects_data = get_defects(session)
            for name, id in defects_data:
                self.defects_combobox.addItem(name, id)

            nn_coefficient_data = get_nn_coeffs(session)

            print(nn_coefficient_data)
        
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
        with self.sessionmaker() as session:
            self.df, self.stage_dict = get_training_data(
                session,
                self.defects_combobox.currentData(),
                self.parameters_model.getCheckedIds(),
                int(self.step_combobox.currentText()),
                self.date_from.dateTime().toPython(),
                self.date_to.dateTime().toPython()
            )

        print(type(self.stage_dict))

        dataset = PolyFilmDataset(
            self.df,
            int(self.window_combobox.currentData()),
            self.stage_dict,
            int(self.step_combobox.currentData()),
        )


    def save_model(self):
        self.df.to_csv('training_data.csv', index=False)
