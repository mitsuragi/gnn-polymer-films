from PySide6.QtWidgets import QComboBox, QSizePolicy

class AdjustedComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Maximum,
                           QSizePolicy.Policy.Preferred)
