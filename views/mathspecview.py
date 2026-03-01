from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QGridLayout, QHBoxLayout, QLabel, QListView, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, QFrame)

from share import ParametersListModel, AdjustedComboBox
from db.db_manager import get_parameters, get_defects, create_engine, get_datetime_range, get_nn_coeffs

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
        parameters_data = get_parameters(self.engine)
        defects_data = get_defects(self.engine)
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
        self.model = ParametersListModel(parameters_data)

        self.parameters_view.setUniformItemSizes(True)
        self.parameters_view.setSelectionMode(QListView.NoSelection)
        self.parameters_view.setModel(self.model)

        forecasting_layout.addWidget(self.forecasting_label, 0, 0)
        forecasting_layout.addWidget(self.defects_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        forecasting_layout.addWidget(self.defects_combobox, 1, 1)
        forecasting_layout.addWidget(self.parameters_label, 2, 0, Qt.AlignmentFlag.AlignRight)
        forecasting_layout.addWidget(self.parameters_view, 2, 1)

        min_dt, max_dt = get_datetime_range(self.engine)

        self.timerange_label = QLabel('Выбор временного диапазона')

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
        date_layout.addWidget(self.timerange_label)
        date_layout.addWidget(QLabel('С:'))
        date_layout.addWidget(self.date_from)
        date_layout.addWidget(QLabel('По:'))
        date_layout.addWidget(self.date_to)
        date_layout.addSpacerItem(QSpacerItem(1000,1))

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.train_label)
        frame_layout.addLayout(settings_layout)
        frame_layout.addLayout(forecasting_layout)
        frame_layout.addLayout(date_layout)
        frame_layout.addSpacerItem(QSpacerItem(1, 1000))

        layout.addWidget(frame)
        self.setLayout(layout)

    def set_comboboxes(self):
        defects_data = get_defects(self.engine)
        for name, id in defects_data:
            self.defects_combobox.addItem(name, id)

        nn_coefficient_data = get_nn_coeffs(self.engine)
        
        for id, value, type_id in nn_coefficient_data:
            if type_id == 1:
                self.batch_combobox.addItem(value, id)
            elif type_id == 3:
                self.epoch_combobox.addItem(value, id)
