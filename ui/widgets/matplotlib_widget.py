import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from sklearn.metrics import precision_recall_curve, auc
 
import matplotlib
matplotlib.use("QtAgg")                          # Qt back-end
 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LIGHT_PARAMS = {
    "figure.facecolor":   "#FFFFFF",
    "axes.facecolor":     "#F8FAFC",
    "axes.edgecolor":     "#E2E8F0",
    "axes.labelcolor":    "#475569",
    "axes.titlecolor":    "#1E293B",
    "axes.grid":          True,
    "grid.color":         "#E2E8F0",
    "grid.linestyle":     "--",
    "grid.alpha":         0.8,
    "xtick.color":        "#64748B",
    "ytick.color":        "#64748B",
    "text.color":         "#334155",
    "legend.facecolor":   "#FFFFFF",
    "legend.edgecolor":   "#E2E8F0",
    "legend.labelcolor":  "#334155",
    "lines.linewidth":    2,
    "figure.dpi":         100,
}

COLORS = ["#2563EB", "#F59E0B", "#16A34A", "#DC2626", "#7C3AED", "#0EA5E9"]

class MatplotlibWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet('background-color: #FFFFFF;')

        with plt.rc_context(LIGHT_PARAMS):
            self.fig = Figure(tight_layout=True, facecolor='#FFFFFF')
            self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet('background-color: #FFFFFF;')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._apply_ax_style(self.ax)

    @staticmethod
    def _apply_ax_style(ax) -> None:
        ax.set_facecolor('#F8FAFC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_color('#CBD5E1')
        ax.tick_params(colors='#64748B', labelsize=9)
        ax.yaxis.label.set_color('#475569')
        ax.xaxis.label.set_color('#475569')
        ax.title.set_color('#1E293B')
        ax.title.set_fontweight('bold')

    def _reset(self) -> None:
        with plt.rc_context(LIGHT_PARAMS):
            self.fig.clear()
            self.fig.set_facecolor('#FFFFFF')
            self.ax = self.fig.add_subplot(111)
        self._apply_ax_style(self.ax)

    def _draw(self) -> None:
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_demo_prediction(self, timestamps, actual, predicted) -> None:
        self._reset()

        t = [ts.timestamp() for ts in timestamps]

        self.ax.plot(t, actual, color=COLORS[0], label='Фактические',
                     linewidth=1, alpha=0.7)
        self.ax.plot(t, predicted, color=COLORS[1], label='Спрогнозированные',
                     linewidth=1, linestyle='--', alpha=0.7)

        self.ax.set_title('Фактические и спрогнозированные значения', fontsize=11)
        self.ax.set_xlabel('Время')
        self.ax.set_ylabel('Показатель качества')
        self.ax.legend(loc='lower right', fontsize=9)

        self._draw()

    def plot_demo_production(self, timestamps, values) -> None:
        self._reset()
        t  = [ts.timestamp() for ts in timestamps]
 
        self.ax.plot(t, values, color=COLORS[0], label="Дефект", lw=1)
 
        self.ax.set_title("Значения дефекта", fontsize=11)
        self.ax.set_xlabel("Дата и время")
        self.ax.set_ylabel("Значение дефекта")
 
        lines1, labels1 = self.ax.get_legend_handles_labels()
        self.ax.legend(lines1, labels1, loc='upper right', fontsize=9)

        self._draw()
 
    def plot_demo_training(self) -> None:
        """
        Placeholder: loss/epoch training graph.
        TODO: replace with real training history emitted by the model trainer.
        """
        self._reset()
        epochs     = np.arange(1, 51)
        train_loss = 0.9 * np.exp(-epochs / 15) + np.random.normal(0, 0.015, 50)
        val_loss   = 0.95 * np.exp(-epochs / 18) + np.random.normal(0, 0.02, 50) + 0.04
 
        self.ax.plot(epochs, train_loss, color=COLORS[0], label="Train Loss", lw=2)
        self.ax.plot(epochs, val_loss,   color=COLORS[1], label="Val Loss",
                     lw=2, linestyle="--")
        self.ax.fill_between(epochs, train_loss, val_loss,
                             alpha=0.07, color=COLORS[0])
 
        self.ax.set_title("Кривая обучения (Loss / Epoch)", fontsize=11)
        self.ax.set_xlabel("Эпохи")
        self.ax.set_ylabel("Loss")
        self.ax.legend(fontsize=9)
        self._draw()

    def plot_bar_chart(self, values) -> None:
        self._reset()

        labels = ['TP', 'TN', 'FP', 'FN']
        colors = [COLORS[2], COLORS[0], COLORS[3], COLORS[1]]

        bars = self.ax.bar(labels, values, color = colors, width=0.6,
                           edgecolor='#FFFFFF', linewidth=1)
        self.ax.bar_label(bars, padding=3, fontsize=9, color='#334155')

        self.ax.set_title('Распределение ответов модели', fontsize=11)
        self.ax.set_xlabel('Тип ответа')
        self.ax.set_ylabel('Количество')
        self.ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        self._draw()

    def plot_pr_auc_chart(self, y_true, y_prob) -> None:
        self._reset()

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        baseline = y_true.sum() / len(y_true) if len(y_true) else 0.0

        self.ax.plot(recall, precision, color=COLORS[0], lw=2,
                     label=f'PR-кривая (AUC = {pr_auc:.3f})')
        self.ax.fill_between(recall, precision, alpha=0.15, color=COLORS[0])
        self.ax.axhline(baseline, color=COLORS[3], lw=1.5, linestyle='--',
                        label=f'Базовый уровень ({baseline:.3f})')

        self.ax.set_title('Precision-Recall кривая', fontsize=11)
        self.ax.set_xlabel('Recall (Полнота)')
        self.ax.set_ylabel('Precision (Точность)')
        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.05)
        self.ax.legend(loc='lower right', fontsize=9)
        
        self._draw()

    def plot_class_distribution(self, y_true, y_prob, threshold) -> None:
        self._reset()

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        probs_negative = y_prob[y_true == 0]
        probs_positive = y_prob[y_true == 1]

        # Реальный диапазон значений по перцентилям — выбросы не растягивают график
        low, high = np.percentile(y_prob, [0.5, 99.5])
        low = min(low, threshold)
        high = max(high, threshold)
        span = high - low if high > low else 0.01
        padding = span * 0.08
        range_min = max(0.0, low - padding)
        range_max = min(1.0, high + padding)

        # Число бинов подстраиваем под объём выборки
        n_bins = int(np.clip(np.sqrt(len(y_prob)), 20, 60))
        bins = np.linspace(range_min, range_max, n_bins + 1)

        # density=True - форма распределения видна независимо от баланса классов
        self.ax.hist(probs_negative, bins=bins, color=COLORS[0], alpha=0.6, density=True,
                     label=f'Класс 0 (негативный), n={len(probs_negative)}',
                     edgecolor='#FFFFFF', linewidth=0.5)
        self.ax.hist(probs_positive, bins=bins, color=COLORS[3], alpha=0.6, density=True,
                     label=f'Класс 1 (позитивный), n={len(probs_positive)}',
                     edgecolor='#FFFFFF', linewidth=0.5)

        self.ax.axvline(threshold, color='#475569', lw=1.5, linestyle=':',
                        label=f'Порог = {threshold:.3f}')

        self.ax.set_title('Распределение предсказанных вероятностей по классам', fontsize=11)
        self.ax.set_xlabel('Предсказанная вероятность')
        self.ax.set_ylabel('Плотность распределения')
        self.ax.set_xlim(range_min, range_max)
        self.ax.legend(fontsize=9)

        self._draw()
