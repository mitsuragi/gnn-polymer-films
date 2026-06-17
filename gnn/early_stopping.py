from pathlib import Path
import torch
from .model import PolymerGAT

class EarlyStopping:
  """
  Останавливает обучение, если val-метрика не улучшается `patience` эпох.
 
  Parameters
  ----------
  patience : int
    Сколько эпох ждать улучшения.
  min_delta : float
    Минимальное изменение, считающееся улучшением.
  mode : str
    "min" для loss/ошибок, "max" для accuracy/f1/recall.
  checkpoint_path : str | Path
    Куда сохранять лучшие веса модели.
  """
  def __init__(
    self,
    patience: int = 10,
    min_delta: float = 1e-4,
    mode: str = 'min',
    checkpoint_path: str | Path = 'best_model.pt',
  ):
    self.patience = patience
    self.min_delta = min_delta
    self.mode = mode
    self.checkpoint_path = checkpoint_path
    self._best_score: float | None = None
    self._counter = 0
    self.triggered = False
    self.last_improved = False

  def step(self, score: float, model: PolymerGAT) -> bool:
    """
    Проверяет улучшение и сохраняет веса при необходимости.
 
    Returns
    -------
    True, если обучение следует продолжать; False — остановить.
    """
    improved = self._is_improved(score)
    self.last_improved = improved

    if improved:
      self._best_score = score
      self._counter = 0
      torch.save(model.state_dict(), self.checkpoint_path)
    else:
      self._counter += 1
      if self._counter >= self.patience:
        self.triggered = True
        return False

    return True

  def _is_improved(self, score: float) -> bool:
    if self._best_score is None:
      return True
    if self.mode == 'min':
      return score < self._best_score - self.min_delta

    return score >= self._best_score + self.min_delta

  @property
  def best_score(self) -> float | None:
    return self._best_score

  @property
  def epochs_without_improvement(self) -> int:
    return self._counter
