from PySide6.QtWidgets import QStackedWidget

class NavigationManager:
    def __init__(self, stack: QStackedWidget):
        self.stack = stack

        self.pages = {}
        self.factories = {}

    def register(self, name, factory):
        self.factories[name] = factory

    def _create_page(self, name):
        if name not in self.factories:
            raise ValueError(f'Page {name} not registered')

        page = self.factories[name](self)

        self.stack.addWidget(page)
        return page

    def get_page(self, name):
        if name not in self.pages:
            self.pages[name] = self._create_page(name)

        return self.pages[name]

    def navigate(self, name, clear_current=True, **kwargs):
        current = self.stack.currentWidget()

        if clear_current and current:
            for key, value in list(self.pages.items()):
                if value == current:
                    self.clear(key)
                    break

        page = self.get_page(name)

        if hasattr(page, 'on_navigate'):
            page.on_navigate(**kwargs)

        self.stack.setCurrentWidget(page)

    def clear(self, name):
        if name in self.pages:
            page = self.pages.pop(name)
            self.stack.removeWidget(page)
            page.deleteLater()
