from torch_geometric.data import Dataset, Data
import torch
import numpy as np
import pandas as pd
from scipy import stats

def pearson_matrix(df: pd.DataFrame) -> np.ndarray:
  """Корреляция Пирсона между всеми парами признаков."""
  return df.corr(method='pearson').values

def spearman_matrix(df: pd.DataFrame) -> np.ndarray:
  """Ранговая корреляция Спирмена (устойчива к выбросам)."""
  return df.corr(method='spearman').values

def mutual_info_matrix(df: pd.DataFrame) -> np.ndarray:
  """
  Нормированная взаимная информация (NMI) через оценку плотности.
  Симметрична, лежит в [0, 1].
  """
  n_features = df.shape[1]
  mi = np.zeros((n_features, n_features))
  arr = df.values.astype(np.float32)
  for i in range(n_features):
    for j in range(i + 1, n_features):
      # kernel-density оценка двумерной MI через корреляцию Спирмена
      # (быстрое приближение без sklearn)
      r, _ = stats.spearmanr(arr[:, i], arr[:, j])
      r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
      nmi = abs(r) # приближение: |r| ≈ NMI для монотонных связей
      mi[i, j] = nmi
      mi[j, i] = nmi

  np.fill_diagonal(mi, 1.0)
  return mi

def partial_corr_matrix(df: pd.DataFrame) -> np.ndarray:
  """
  Частичная корреляция через обращение корреляционной матрицы.
  Показывает прямую связь между парой переменных при фиксации остальных.
  """
  corr = df.corr(method='pearson').values
  try:
    inv = np.linalg.pinv(corr)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return np.abs(partial)
  except np.linalg.LinAlgError:
    return np.abs(corr)

EDGE_STRATEGIES = {
  'pearson':      pearson_matrix,
  'spearman':     spearman_matrix,
  'mutual_info':  mutual_info_matrix,
  'partial_corr': partial_corr_matrix,
}

class PolyFilmDataset(Dataset):
  def __init__(
    self,
    df: pd.DataFrame,
    target_col: str = 'target',
    edge_strategy: str | list[str] = 'pearson',
    threshold: float = 0.3,
    self_loops: bool = True,
    normalize_features: bool = True,
  ):
    super().__init__()

    self._feature_cols: list[str] = [c for c in df.columns if c != target_col]
    self._n_nodes: int = len(self._feature_cols)

    feature_df = df[self._feature_cols].copy().astype(np.float32)
    labels = df[target_col].values.astype(np.int64)

    if normalize_features:
      self._mean = feature_df.mean()
      self._std = feature_df.std().replace(0, 1)
      feature_df = (feature_df - self._mean) / self._std
    else:
      self._mean = None
      self._std = None

    self._features: np.ndarray = feature_df.values
    self._labels: np.ndarray = labels

    strategies = [edge_strategy] if isinstance(edge_strategy, str) else edge_strategy
    for s in strategies:
      if s not in EDGE_STRATEGIES:
        raise ValueError(
          f"Неизвестная стратегия '{s}'. "
          f"Доступные: {list(EGDE_STRATEGIES)}"
        )

    weight_matrices = [
      np.abs(EDGE_STRATEGIES[s](feature_df))
      for s in strategies
    ]

    combined: np.ndarray = np.mean(weight_matrices, axis=0)
    np.fill_diagonal(combined, 0.0)

    src, dst = np.where(combined >= threshold)
    edge_weights = combined[src, dst].astype(np.float32)

    if self_loops:
      loop_idx = np.arange(self._n_nodes)
      src = np.concatenate([src, loop_idx])
      dst = np.concatenate([dst, loop_idx])
      edge_weights = np.concatenate([edge_weights, np.ones(self._n_nodes)])

    self._edge_index: torch.Tensor = torch.tensor(
      np.stack([src, dst], axis=0), dtype=torch.long
    )
    self._edge_attr: torch.Tensor = torch.tensor(
      edge_weights,  dtype=torch.float32
    ).unsqueeze(1)

  def len(self) -> int:
    return len(self._labels)

  def get(self, idx: int) -> Data:
    """
    Возвращает граф для одного наблюдения.
 
    Граф:
    x          — (N_nodes, 1)  значения признаков для данного наблюдения
    edge_index — (2, E)        индексы рёбер (общие для всех графов)
    edge_attr  — (E, 1)        веса рёбер (общие для всех графов)
    y          — (1,)          бинарная метка
    """
    
    # Значения признаков текущего наблюдения — каждый узел получает своё число
    node_features = torch.tensor(
      self._features[idx], dtype=torch.float32
    ).unsqueeze(1)

    label = torch.tensor(self._labels[idx], dtype=torch.long)

    return Data(
      x             = node_features,
      edge_index    = self._edge_index,
      edge_attr     = self._edge_attr,
      y             = label,
      num_nodes     = self._n_nodes,
    )

  @property
  def num_node_features(self) -> int:
    return 1

  @property
  def num_classes(self) -> int:
    return 2

  @property
  def feature_names(self) -> list[str]:
    return self._feature_cols

  @property
  def num_edges(self) -> int:
    return self._edge_index.shape[1]

  def __repr__(self) -> str:
    return (
      f"{self.__class__.__name__}("
      f"samples={self.len()}, "
      f"nodes={self._n_nodes}, "
      f"edges={self.num_edges})"
    )
