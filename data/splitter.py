from typing import NamedTuple
import pandas as pd 
import numpy as np 
from .dataset import PolyFilmDataset

class SplitResult(NamedTuple):
  train: pd.DataFrame
  val: pd.DataFrame
  test: pd.DataFrame

def stratified_split(
  df: pd.DataFrame,
  target_col: str = 'target',
  train_size: float = 0.70,
  val_size: float = 0.15,
  test_size: float = 0.15,
  random_state: int = 42,
  verbose: bool = True,
) -> SplitResult:
  """
  Делит DataFrame на train / val / test с сохранением доли классов.
 
  Parameters
  ----------
  df : pd.DataFrame
      Исходный датафрейм со всеми признаками и целевой переменной.
  target_col : str
      Название столбца с бинарной меткой (0 / 1).
  train_size, val_size, test_size : float
      Доли разбиения. Должны в сумме давать 1.0.
  random_state : int
      Зерно генератора для воспроизводимости.
  verbose : bool
      Печатать ли статистику по сплитам.
 
  Returns
  -------
  SplitResult(train, val, test) — три DataFrame с reset_index.
  """
  if not np.isclose(train_size + val_size + test_size, 1.0):
    raise ValueError(
      f'Сумма долей должна быть 1.0'
      f'получено {train_size + val_size + test_size:.4f}'
    )

  if target_col not in df.columns:
    raise KeyError(f"Стобец '{target_col}' не найден в DataFrame")

  unique_classes = df[target_col].unique()
  if not set(unique_classes).issubset({0.0, 1.0}):
    raise ValueError(f"Ожидаются только значения 0 и 1 в '{target_col}'")

  rng = np.random.default_rng(random_state)

  idx_pos = df.index[df[target_col] == 1].to_numpy().copy()
  idx_neg = df.index[df[target_col] == 0].to_numpy().copy()

  rng.shuffle(idx_pos)
  rng.shuffle(idx_neg)

  def _split_indices(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Режет массив индексов на три части по заданным пропорциям."""
    n = len(idx)
    n_train = int(np.floor(n * train_size))
    n_val = int(np.floor(n * val_size))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

  train_pos, val_pos, test_pos = _split_indices(idx_pos)
  train_neg, val_neg, test_neg = _split_indices(idx_neg)

  def _make_split(pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
    combined = np.concatenate([pos, neg])
    rng.shuffle(combined)
    return df.loc[combined].reset_index(drop=True)

  train_df = _make_split(train_pos, train_neg)
  val_df = _make_split(val_pos, val_neg)
  test_df = _make_split(test_pos, test_neg)

  if verbose:
    _print_split_stats(df, train_df, val_df, test_df, target_col)

  return SplitResult(train=train_df, val=val_df, test=test_df)

def _print_split_stats(
  original: pd.DataFrame,
  train: pd.DataFrame,
  val: pd.DataFrame,
  test: pd.DataFrame,
  target_col: str,
):
  total = len(original)

  header = f"{'Сплит':<10} {'Строк':>7} {'% от всех':>10} {'Дефекты':>9} {'% дефектов':>12}"
  sep = '─' * len(header)

  print(sep)
  print(header)
  print(sep)

  for name, df in [('Исходный', original), ('Train', train), ('Val', val), ('Test', test)]:
    n = len(df)
    pct = 100 * n / total
    n_pos = (df[target_col] == 1).sum()
    pct_pos = 100 * n_pos / n if n > 0 else 0.0
    print(f"{name:<10} {n:>7,}  {pct:>9.1f}%  {n_pos:>8,}  {pct_pos:>10.1f}%")

    print(sep)
