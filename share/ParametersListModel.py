from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QPersistentModelIndex

class ParametersListModel(QAbstractListModel):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.checked = set()
        # self.items = []

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        param_id, param_name = self.items[row]

        if role == Qt.DisplayRole:
            return param_name

        if role == Qt.CheckStateRole:
            if param_id in self.checked:
                return Qt.Checked
            return Qt.Unchecked

        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex, /) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        return (
            Qt.ItemFlag.ItemIsEnabled |
            Qt.ItemFlag.ItemIsSelectable |
            Qt.ItemFlag.ItemIsUserCheckable
        )

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False

        if role == Qt.CheckStateRole:
            param_id = self.items[index.row()][0]

            if value:
                self.checked.add(param_id)
            else:
                self.checked.discard(param_id)

            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True

        return False

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def setItemsList(self, new_items):
        self.beginResetModel()
        self.items = new_items
        self.checked.clear()
        self.endResetModel()

    def getCheckedIds(self):
        return list(self.checked)
