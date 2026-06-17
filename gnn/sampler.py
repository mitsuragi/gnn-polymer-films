import numpy as np 
from torch.utils.data import WeightedRandomSampler
import torch

def make_weighted_sampler(
  labels: np.ndarray,
  pos_ratio: float = 0.3,
) -> WeightedRandomSampler:
  """
  Создаёт сэмплер, который в среднем даёт `pos_ratio` дефектов в батче.

  Parameters
  ----------
  labels    : бинарный массив меток (0/1) обучающей выборки
  pos_ratio : желаемая доля дефектов в батче
  """
  n = len(labels)
  n_pos = labels.sum()
  n_neg = n - n_pos

  if n_pos == 0:
    raise ValueError("В обучающей выборке нет ни одного дефекта")

  w_pos = pos_ratio / n_pos
  w_neg = (1.0 - pos_ratio) / n_neg
  weights = np.where(labels == 1, w_pos, w_neg)

  return WeightedRandomSampler(
    weights= torch.tensor(weights, dtype=torch.float64),
    num_samples = n,
    replacement = True,
  )

def make_event_aware_sampler(
  labels: np.ndarray,
  pos_ratio: float = 0.15,
  context_radius: int = 20,
  hard_neg_ratio: float = 0.50,
) -> WeightedRandomSampler:
  """
  Сэмплер для временного ряда с редкими событиями.

  В батч чаще попадают:
  1. сами дефекты;
  2. отрицательные точки рядом с дефектом — сложные near-event negative;
  3. остальные отрицательные точки.
  """
  labels = np.asarray(labels).astype(int)
  n = len(labels)
  pos_idx = np.flatnonzero(labels == 1)
  neg_idx = np.flatnonzero(labels == 0)

  if len(pos_idx) == 0:
    raise ValueError('В обучающей выборке нет ни одного дефекта')
  if len(neg_idx) == 0:
    raise ValueError('В обучающей выборке нет отрицательных примеров')

  context_mask = np.zeros(n, dtype=bool)
  for idx in pos_idx:
    left = max(0, idx - context_radius)
    right = min(n, idx + context_radius + 1)
    context_mask[left:right] = True
  context_mask &= labels == 0

  context_idx = np.flatnonzero(context_mask)
  easy_neg_idx = np.flatnonzero((labels == 0) & ~context_mask)

  weights = np.zeros(n, dtype=np.float64)
  weights[pos_idx] = pos_ratio / len(pos_idx)

  neg_mass = 1.0 - pos_ratio
  if len(context_idx) > 0 and len(easy_neg_idx) > 0:
    weights[context_idx] = (neg_mass * hard_neg_ratio) / len(context_idx)
    weights[easy_neg_idx] = (neg_mass * (1.0 - hard_neg_ratio)) / len(easy_neg_idx)
  elif len(context_idx) > 0:
    weights[context_idx] = neg_mass / len(context_idx)
  else:
    weights[easy_neg_idx] = neg_mass / len(easy_neg_idx)

  return WeightedRandomSampler(
    weights=torch.tensor(weights, dtype=torch.float64),
    num_samples=n,
    replacement=True,
  )
