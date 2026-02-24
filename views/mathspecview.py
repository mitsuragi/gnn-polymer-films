from PySide6.QtGui import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QSpacerItem, QVBoxLayout, QWidget, QFrame)

class MathSpecialistView(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Интерфейс специалиста по математическому обеспечению')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(15, 10, 15, 10)
        frame_layout.setSpacing(10)

        self.label = QLabel('Интерфейс специалиста по математическому обеспечению')
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.train_label = QLabel('Обучение модели')
        self.train_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(6) 

        self.layers_label = QLabel('Количество скрытых слоев')
        self.layers_combobox = QComboBox()
        self.batch_label = QLabel('Размер батча')
        self.batch_combobox = QComboBox()
        self.epoch_label = QLabel('Максимальное количество эпох')
        self.epoch_combobox = QComboBox()

        settings_layout.addWidget(self.layers_label)
        settings_layout.addWidget(self.layers_combobox)
        settings_layout.addWidget(self.batch_label)
        settings_layout.addWidget(self.batch_combobox)
        settings_layout.addWidget(self.epoch_label)
        settings_layout.addWidget(self.epoch_combobox)

        settings_layout.addSpacerItem(QSpacerItem(1000, 1))

        frame_layout.addWidget(self.label)
        frame_layout.addWidget(self.train_label)
        frame_layout.addLayout(settings_layout)
        frame_layout.addSpacerItem(QSpacerItem(1, 1000))

        layout.addWidget(frame)
        self.setLayout(layout)
