from __future__ import annotations
 
import numpy as np
import pandas as pd
 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
 
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox,
    QCheckBox, QFrame, QSizePolicy, QSpacerItem,
    QButtonGroup, QToolButton,
)
 
# from gnn import PredictionResult

PALETTE = {
    "bg":           "#1C1C1E",
    "bg_card":      "#2C2C2E",
    "bg_panel":     "#252527",
    "accent_blue":  "#3D85F7",
    "accent_green": "#30D158",
    "accent_red":   "#FF453A",
    "accent_amber": "#FFD60A",
    "text_primary": "#F2F2F7",
    "text_secondary": "#8E8E93",
    "border":       "#3A3A3C",
    "true_pos":     "#30D158",   # TP — зелёный
    "true_neg":     "#3D85F7",   # TN — синий
    "false_pos":    "#FFD60A",   # FP — жёлтый
    "false_neg":    "#FF453A",   # FN — красный
}
 
_MPL_STYLE = {
    "figure.facecolor":     PALETTE["bg"],
    "axes.facecolor":       PALETTE["bg_card"],
    "axes.edgecolor":       PALETTE["border"],
    "axes.labelcolor":      PALETTE["text_secondary"],
    "axes.titlecolor":      PALETTE["text_primary"],
    "axes.grid":            True,
    "grid.color":           PALETTE["border"],
    "grid.linewidth":       0.6,
    "grid.alpha":           0.8,
    "xtick.color":          PALETTE["text_secondary"],
    "ytick.color":          PALETTE["text_secondary"],
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "text.color":           PALETTE["text_primary"],
    "legend.facecolor":     PALETTE["bg_panel"],
    "legend.edgecolor":     PALETTE["border"],
    "legend.labelcolor":    PALETTE["text_primary"],
    "legend.fontsize":      8,
    "lines.antialiased":    True,
    "font.family":          "monospace",
}

class PredictionCanvas(FigureCanvas):
    """
    Matplotlib-полотно с тремя связанными осями:
      1. Непрерывная вероятность дефекта P(дефект) + порог
      2. Дискретные предсказания vs истинные метки (scatter)
      3. Расхождения (TP / TN / FP / FN) в виде цветных полос
    """
 
    def __init__(self, parent: QWidget | None = None) -> None:
        import matplotlib as mpl
        mpl.rcParams.update(_MPL_STYLE)
 
        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
 
        self._ax_prob:  object | None = None
        self._ax_cls:   object | None = None
        self._ax_diff:  object | None = None
 
        self._setup_axes()
 
    def _setup_axes(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(
            2, 1,
            height_ratios=[3, 2],
            hspace=0.08,
        )
        self._ax_prob = self.fig.add_subplot(gs[0])
        self._ax_cls  = self.fig.add_subplot(gs[1], sharex=self._ax_prob)
        # self._ax_diff = self.fig.add_subplot(gs[2], sharex=self._ax_prob)
 
        for ax in (self._ax_prob, self._ax_cls):
            ax.tick_params(labelbottom=False)
 
        self._ax_prob.set_ylabel("P(дефект)", labelpad=8)
        self._ax_cls.set_ylabel("Класс", labelpad=8)
        # self._ax_diff.set_ylabel("Разница", labelpad=8)
        # self._ax_diff.set_xlabel("Наблюдение", labelpad=8)
 
    # ── Основной метод перерисовки ────────────────────────────────────────────
 
    def plot(
        self,
        y_true:    np.ndarray,
        y_pred:    np.ndarray,
        y_prob:    np.ndarray,
        threshold: float = 0.5,
        x_index:   np.ndarray | None = None,
        show_fill: bool = True,
        window:    tuple[int, int] | None = None,
    ) -> None:
        """
        Перерисовывает все три оси.
 
        Parameters
        ----------
        y_true    : (N,) int   — истинные метки
        y_pred    : (N,) int   — предсказанные метки
        y_prob    : (N,) float — вероятности дефекта
        threshold : float       — порог классификации
        x_index   : (N,) array — метки оси X (индексы, метки времени и т.д.)
        show_fill : bool        — закрашивать области под кривой вероятности
        window    : (start, end) — отображать только срез [start:end]
        """
        n = len(y_true)
        x = x_index if x_index is not None else np.arange(n)
 
        # Применяем окно просмотра
        if window is not None:
            s, e = max(0, window[0]), min(n, window[1])
            y_true = y_true[s:e]; y_pred = y_pred[s:e]
            y_prob = y_prob[s:e]; x      = x[s:e]
 
        self._setup_axes()
 
        # ── Ось 1: вероятность ────────────────────────────────────────────────
        ax = self._ax_prob
 
        if show_fill:
            ax.fill_between(x, y_prob, threshold,
                            where=(y_prob >= threshold),
                            alpha=0.18, color=PALETTE["accent_red"],
                            interpolate=True)
            ax.fill_between(x, y_prob, threshold,
                            where=(y_prob < threshold),
                            alpha=0.12, color=PALETTE["accent_blue"],
                            interpolate=True)
 
        ax.plot(x, y_prob, color=PALETTE["accent_blue"],
                linewidth=1.4, zorder=3, label="P(дефект)")
        ax.axhline(threshold, color=PALETTE["accent_amber"],
                   linewidth=1.1, linestyle="--", zorder=4,
                   label=f"Порог {threshold:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.legend(loc="upper right", framealpha=0.85)
        ax.set_title("Прогноз дефектов — GAT модель", fontsize=11, pad=10)
 
        # ── Ось 2: классы — scatter ───────────────────────────────────────────
        ax2 = self._ax_cls
        jitter = np.random.default_rng(0).uniform(-0.07, 0.07, size=len(x))
 
        tp = (y_true == 1) & (y_pred == 1)
        tn = (y_true == 0) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
 
        scatter_cfg = [
            (y_true, tp, PALETTE["true_pos"],  "o", "TP (верный дефект)",   60),
            (y_true, tn, PALETTE["true_neg"],  "o", "TN (верная норма)",    40),
            (y_true, fp, PALETTE["false_pos"], "X", "FP (ложная тревога)",  70),
            (y_true, fn, PALETTE["false_neg"], "X", "FN (пропуск дефекта)", 70),
        ]
        for yt, mask, color, marker, label, size in scatter_cfg:
            if mask.any():
                ax2.scatter(x[mask], yt[mask] + jitter[mask],
                            c=color, marker=marker, s=size,
                            alpha=0.85, zorder=4, label=label,
                            linewidths=0.4, edgecolors="none")
 
        # Линия предсказания
        ax2.step(x, y_pred, where="post",
                 color=PALETTE["accent_amber"], linewidth=1.0,
                 alpha=0.6, zorder=3, label="Предсказание")
 
        ax2.set_ylim(-0.4, 1.4)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Норма (0)", "Дефект (1)"], fontsize=8)
        ax2.legend(loc="upper right", framealpha=0.85, ncol=2)
 
 #        # ── Ось 3: расхождения ────────────────────────────────────────────────
 #        ax3 = self._ax_diff
 #        diff = (y_pred.astype(int) - y_true.astype(int)).astype(float)
 #        colors_diff = np.where(diff > 0, PALETTE["false_pos"],
 #                      np.where(diff < 0, PALETTE["false_neg"],
 #                               PALETTE["border"]))
 # 
 #        ax3.bar(x, diff, color=colors_diff, width=0.8, zorder=3)
 #        ax3.axhline(0, color=PALETTE["text_secondary"],
 #                    linewidth=0.8, zorder=4)
 #        ax3.set_ylim(-1.5, 1.5)
 #        ax3.set_yticks([-1, 0, 1])
 #        ax3.set_yticklabels(["FN", "0", "FP"], fontsize=7)
 
        self.draw_idle()

class ControlPanel(QFrame):
    """Панель с элементами управления графиком."""
 
    threshold_changed = Signal(float)
    window_changed    = Signal(int, int)
    fill_toggled      = Signal(bool)
    zoom_reset        = Signal()
    export_requested  = Signal()
 
    def __init__(self, n_total: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.n_total = n_total
        self._build_ui()
        self._apply_style()
 
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)
 
        # ── Порог классификации ───────────────────────────────────────────────
        layout.addWidget(self._label("Порог:"))
 
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(1, 99)
        self.slider_threshold.setValue(50)
        self.slider_threshold.setFixedWidth(120)
        self.slider_threshold.setToolTip("Порог классификации (0.01 – 0.99)")
        self.slider_threshold.valueChanged.connect(self._on_threshold)
        layout.addWidget(self.slider_threshold)
 
        self.lbl_threshold = QLabel("0.50")
        self.lbl_threshold.setFixedWidth(34)
        layout.addWidget(self.lbl_threshold)
 
        layout.addWidget(self._separator())
 
        # ── Окно просмотра ────────────────────────────────────────────────────
        layout.addWidget(self._label("Окно:"))
 
        self.combo_window = QComboBox()
        self.combo_window.addItems(["Все", "50", "100", "200", "500"])
        self.combo_window.setFixedWidth(70)
        self.combo_window.currentTextChanged.connect(self._on_window)
        layout.addWidget(self.combo_window)
 
        layout.addWidget(self._separator())
 
        # ── Заливка ───────────────────────────────────────────────────────────
        self.chk_fill = QCheckBox("Заливка")
        self.chk_fill.setChecked(True)
        self.chk_fill.toggled.connect(self.fill_toggled.emit)
        layout.addWidget(self.chk_fill)
 
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))
 
        # ── Кнопки ────────────────────────────────────────────────────────────
        btn_reset = self._button("⟳ Сброс", self.zoom_reset.emit)
        btn_export = self._button("↓ Экспорт", self.export_requested.emit)
        layout.addWidget(btn_reset)
        layout.addWidget(btn_export)
 
    def _on_threshold(self, value: int) -> None:
        thr = value / 100.0
        self.lbl_threshold.setText(f"{thr:.2f}")
        self.threshold_changed.emit(thr)
 
    def _on_window(self, text: str) -> None:
        if text == "Все":
            self.window_changed.emit(0, self.n_total)
        else:
            size = int(text)
            start = max(0, self.n_total - size)
            self.window_changed.emit(start, self.n_total)
 
    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 12px;")
        return lbl
 
    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {PALETTE['border']};")
        sep.setFixedHeight(20)
        return sep
 
    def _button(self, text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(slot)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {PALETTE['bg_panel']};
                color: {PALETTE['text_primary']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 0 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {PALETTE['accent_blue']};
                border-color: {PALETTE['accent_blue']};
            }}
            QPushButton:pressed {{
                background: #2A6CD4;
            }}
        """)
        return btn
 
    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            ControlPanel {{
                background: {PALETTE['bg_panel']};
                border-top: 1px solid {PALETTE['border']};
            }}
            QCheckBox {{
                color: {PALETTE['text_primary']};
                font-size: 12px;
            }}
            QComboBox {{
                background: {PALETTE['bg_card']};
                color: {PALETTE['text_primary']};
                border: 1px solid {PALETTE['border']};
                border-radius: 5px;
                padding: 2px 6px;
                font-size: 12px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {PALETTE['border']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px;
                background: {PALETTE['accent_blue']};
                border-radius: 7px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {PALETTE['accent_blue']};
                border-radius: 2px;
            }}
        """)

class StatsBar(QFrame):
    """Горизонтальная полоса с ключевыми метриками."""
 
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(16, 6, 16, 6)
        self._layout.setSpacing(24)
        self._widgets: dict[str, QLabel] = {}
        self._build()
        self.setStyleSheet(f"""
            StatsBar {{
                background: {PALETTE['bg_panel']};
                border-bottom: 1px solid {PALETTE['border']};
            }}
        """)
 
    def _build(self) -> None:
        stats = [
            ("total",     "Всего",      PALETTE["text_secondary"]),
            ("defects",   "Дефектов",   PALETTE["accent_red"]),
            ("precision", "Precision",  PALETTE["accent_green"]),
            ("recall",    "Recall",     PALETTE["accent_green"]),
            ("f1",        "F1",         PALETTE["accent_blue"]),
            ("threshold", "Порог",      PALETTE["accent_amber"]),
        ]
        for key, label, color in stats:
            col = QVBoxLayout()
            col.setSpacing(2)
 
            lbl_name = QLabel(label)
            lbl_name.setStyleSheet(
                f"color: {PALETTE['text_secondary']}; font-size: 10px;"
            )
 
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(
                f"color: {color}; font-size: 15px; font-weight: 700;"
            )
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
 
            col.addWidget(lbl_name, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl_val,  alignment=Qt.AlignmentFlag.AlignCenter)
            self._layout.addLayout(col)
            self._widgets[key] = lbl_val
 
        self._layout.addStretch()
 
    def update_stats(self, metrics: dict | None, n: int,
                     n_defects: int, threshold: float) -> None:
        self._widgets["total"].setText(f"{n:,}")
        pct = 100 * n_defects / n if n else 0
        self._widgets["defects"].setText(f"{n_defects} ({pct:.1f}%)")
        self._widgets["threshold"].setText(f"{threshold:.2f}")
 
        if metrics:
            self._widgets["precision"].setText(f"{metrics.get('precision', 0):.3f}")
            self._widgets["recall"].setText(f"{metrics.get('recall', 0):.3f}")
            self._widgets["f1"].setText(f"{metrics.get('f1', 0):.3f}")
        else:
            for k in ("precision", "recall", "f1"):
                self._widgets[k].setText("n/a")

class PredictionChartWidget(QWidget):
    """
    Виджет для встраивания в основное окно приложения.
 
    Использование
    -------------
    chart = PredictionChartWidget()
    layout.addWidget(chart)                    # вставляем в любой layout
 
    chart.load(result=result, df=df_with_target, target_col="target")
    """
 
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result:    PredictionResult | None = None
        self._y_true:    np.ndarray | None       = None
        self._threshold: float                   = 0.5
        self._window:    tuple[int, int]         = (0, 0)
        self._show_fill: bool                    = True
        self._build_ui()
        self._apply_widget_style()
 
    # ── Публичный API ─────────────────────────────────────────────────────────
 
    def load(
        self,
        result:     PredictionResult,
        df:         pd.DataFrame,
        target_col: str = "target",
    ) -> None:
        """
        Загружает данные и перерисовывает график.
 
        Parameters
        ----------
        result     : PredictionResult из predictor.predict(df)
        df         : исходный DataFrame с колонкой target_col
        target_col : название колонки с истинными метками
        """
        self._result    = result
        self._threshold = result.defect_threshold
        self._y_true    = df[target_col].values.astype(int) \
                          if target_col in df.columns else None
        n = len(result.predictions)
        self._window = (0, n)
 
        # Синхронизируем слайдер порога
        self._controls.slider_threshold.setValue(int(self._threshold * 100))
        self._controls.n_total = n
 
        self._refresh_stats()
        self._redraw()
 
    # ── Построение UI ─────────────────────────────────────────────────────────
 
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
 
        # Строка статистики
        self._stats_bar = StatsBar()
        root.addWidget(self._stats_bar)
 
        # Canvas
        self._canvas = PredictionCanvas()
        root.addWidget(self._canvas, stretch=1)
 
        # Тулбар matplotlib (zoom, pan, save)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setStyleSheet(f"""
            QToolBar {{
                background: {PALETTE['bg_panel']};
                border: none;
                border-top: 1px solid {PALETTE['border']};
                spacing: 4px;
                padding: 2px 8px;
            }}
            QToolButton {{
                color: {PALETTE['text_secondary']};
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 3px;
            }}
            QToolButton:hover {{
                background: {PALETTE['border']};
                color: {PALETTE['text_primary']};
            }}
        """)
        root.addWidget(self._toolbar)
 
        # Панель управления
        self._controls = ControlPanel(n_total=0)
        self._controls.threshold_changed.connect(self._on_threshold)
        self._controls.window_changed.connect(self._on_window)
        self._controls.fill_toggled.connect(self._on_fill)
        self._controls.zoom_reset.connect(self._on_zoom_reset)
        self._controls.export_requested.connect(self._on_export)
        root.addWidget(self._controls)
 
    def _apply_widget_style(self) -> None:
        self.setStyleSheet(f"background: {PALETTE['bg']};")
 
    # ── Слоты ─────────────────────────────────────────────────────────────────
 
    def _on_threshold(self, value: float) -> None:
        self._threshold = value
        self._refresh_stats()
        self._redraw()
 
    def _on_window(self, start: int, end: int) -> None:
        self._window = (start, end)
        self._redraw()
 
    def _on_fill(self, checked: bool) -> None:
        self._show_fill = checked
        self._redraw()
 
    def _on_zoom_reset(self) -> None:
        self._canvas._ax_prob.autoscale()
        self._canvas.draw_idle()
 
    def _on_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "prediction_chart.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self._canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
 
    # ── Внутренние методы ─────────────────────────────────────────────────────
 
    def _redraw(self) -> None:
        if self._result is None:
            return
 
        y_prob = self._result.probabilities
        y_pred = (y_prob >= self._threshold).astype(int)
        y_true = self._y_true if self._y_true is not None else y_pred.copy()
 
        self._canvas.plot(
            y_true    = y_true,
            y_pred    = y_pred,
            y_prob    = y_prob,
            threshold = self._threshold,
            show_fill = self._show_fill,
            window    = self._window if self._window[1] > 0 else None,
        )
 
    def _refresh_stats(self) -> None:
        if self._result is None:
            return
        y_prob = self._result.probabilities
        y_pred = (y_prob >= self._threshold).astype(int)
        n      = len(y_pred)
        n_def  = int(y_pred.sum())
 
        self._stats_bar.update_stats(None, n, n_def, self._threshold)

        metrics = None
        if self._y_true is not None:
            from gnn.predictor import _compute_metrics
            metrics = _compute_metrics(self._y_true, y_pred, y_prob)
 
        self._stats_bar.update_stats(metrics, n, n_def, self._threshold)
