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

def temporal_split(
  df: pd.DataFrame,
  target_col: str = 'target',
  time_col: str = 'timestamp',
  train_size: float = 0.70,
  val_size: float = 0.15,
  test_size: float = 0.15,
  purge_gap: int = 0,
  verbose: bool = True,
) -> SplitResult:
  """
  Делит временной ряд по времени: train — прошлое, val/test — будущее.

  purge_gap — сколько строк выбросить между сплитами, чтобы соседние точки
  не протекали в оценку при использовании временных окон или лагов.
  """
  if not np.isclose(train_size + val_size + test_size, 1.0):
    raise ValueError(
      f'Сумма долей должна быть 1.0, получено {train_size + val_size + test_size:.4f}'
    )
  if target_col not in df.columns:
    raise KeyError(f"Столбец '{target_col}' не найден в DataFrame")
  if time_col not in df.columns:
    raise KeyError(f"Столбец времени '{time_col}' не найден в DataFrame")

  ordered = df.sort_values(time_col).reset_index(drop=True)
  n = len(ordered)
  train_end = int(np.floor(n * train_size))
  val_end = train_end + int(np.floor(n * val_size))

  val_start = min(train_end + purge_gap, n)
  test_start = min(val_end + purge_gap, n)

  train_df = ordered.iloc[:train_end].reset_index(drop=True)
  val_df = ordered.iloc[val_start:val_end].reset_index(drop=True)
  test_df = ordered.iloc[test_start:].reset_index(drop=True)

  if verbose:
    _print_temporal_split_stats(ordered, train_df, val_df, test_df, target_col, time_col, purge_gap)

  return SplitResult(train=train_df, val=val_df, test=test_df)

def _print_temporal_split_stats(
  original: pd.DataFrame,
  train: pd.DataFrame,
  val: pd.DataFrame,
  test: pd.DataFrame,
  target_col: str,
  time_col: str,
  purge_gap: int,
):
  total = len(original)
  header = f"{'Сплит':<10} {'Строк':>7} {'% от всех':>10} {'Дефекты':>9} {'% дефектов':>12} {'Период':>28}"
  sep = '-' * len(header)
  print(sep)
  print(header)
  print(sep)

  for name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
    n = len(split_df)
    pct = 100 * n / total if total > 0 else 0.0
    n_pos = int((split_df[target_col] == 1).sum()) if n > 0 else 0
    pct_pos = 100 * n_pos / n if n > 0 else 0.0
    if n > 0:
      period = f"{split_df[time_col].iloc[0]} -> {split_df[time_col].iloc[-1]}"
    else:
      period = 'empty'
    print(f"{name:<10} {n:>7,}  {pct:>9.1f}%  {n_pos:>8,}  {pct_pos:>10.2f}%  {period:>28}")

  if purge_gap > 0:
    print(f"purge_gap: {purge_gap} строк между соседними сплитами")
  print(sep)


def temporal_event_split(
  df: pd.DataFrame,
  target_col: str = 'target',
  time_col: str = 'timestamp',
  train_event_size: float = 0.70,
  val_event_size: float = 0.15,
  test_event_size: float = 0.15,
  purge_gap: int = 0,
  verbose: bool = True,
) -> SplitResult:
  """
  Хронологический split для редких событий.

  Границы выбираются не по доле строк, а по доле positive-событий:
  первые события идут в train, следующие — в val, последние — в test.
  Так val/test остаются будущими периодами и при этом не оказываются без дефектов.
  """
  if not np.isclose(train_event_size + val_event_size + test_event_size, 1.0):
    raise ValueError(
      'Сумма долей событий должна быть 1.0, '
      f'получено {train_event_size + val_event_size + test_event_size:.4f}'
    )
  if target_col not in df.columns:
    raise KeyError(f"Столбец '{target_col}' не найден в DataFrame")
  if time_col not in df.columns:
    raise KeyError(f"Столбец времени '{time_col}' не найден в DataFrame")

  ordered = df.sort_values(time_col).reset_index(drop=True)
  pos_idx = np.flatnonzero(ordered[target_col].values == 1)
  n_pos = len(pos_idx)
  if n_pos < 3:
    raise ValueError('Для temporal_event_split нужно хотя бы 3 positive-события')

  train_pos_end = max(1, int(np.floor(n_pos * train_event_size)))
  val_pos_end = max(train_pos_end + 1, int(np.floor(n_pos * (train_event_size + val_event_size))))
  val_pos_end = min(val_pos_end, n_pos - 1)

  train_end = int(pos_idx[train_pos_end])
  val_end = int(pos_idx[val_pos_end])

  val_start = min(train_end + purge_gap, len(ordered))
  test_start = min(val_end + purge_gap, len(ordered))

  train_df = ordered.iloc[:train_end].reset_index(drop=True)
  val_df = ordered.iloc[val_start:val_end].reset_index(drop=True)
  test_df = ordered.iloc[test_start:].reset_index(drop=True)

  if verbose:
    print(
      f"event split: train_pos≈{train_event_size:.0%}, "
      f"val_pos≈{val_event_size:.0%}, test_pos≈{test_event_size:.0%}"
    )
    _print_temporal_split_stats(ordered, train_df, val_df, test_df, target_col, time_col, purge_gap)

  return SplitResult(train=train_df, val=val_df, test=test_df)


def _positive_episodes(labels: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
  pos_idx = np.flatnonzero(np.asarray(labels).astype(int) == 1)
  if len(pos_idx) == 0:
    return []

  episodes = []
  start = prev = int(pos_idx[0])
  for idx in pos_idx[1:]:
    idx = int(idx)
    if idx - prev <= max_gap:
      prev = idx
    else:
      episodes.append((start, prev))
      start = prev = idx
  episodes.append((start, prev))
  return episodes

def temporal_episode_split(
  df: pd.DataFrame,
  target_col: str = 'target',
  time_col: str = 'timestamp',
  train_episode_size: float = 0.70,
  val_episode_size: float = 0.15,
  test_episode_size: float = 0.15,
  event_max_gap: int = 50,
  purge_gap: int = 0,
  verbose: bool = True,
) -> SplitResult:
  """
  Хронологический split по эпизодам редких событий.

  Positive-точки, расстояние между которыми <= event_max_gap строк, считаются
  одним эпизодом. Границы train/val/test ставятся между эпизодами, чтобы
  будущие сплиты содержали целые дефектные эпизоды и окружающие negative-точки.
  """
  if not np.isclose(train_episode_size + val_episode_size + test_episode_size, 1.0):
    raise ValueError(
      'Сумма долей эпизодов должна быть 1.0, '
      f'получено {train_episode_size + val_episode_size + test_episode_size:.4f}'
    )
  if target_col not in df.columns:
    raise KeyError(f"Столбец '{target_col}' не найден в DataFrame")
  if time_col not in df.columns:
    raise KeyError(f"Столбец времени '{time_col}' не найден в DataFrame")

  ordered = df.sort_values(time_col).reset_index(drop=True)
  episodes = _positive_episodes(ordered[target_col].values, max_gap=event_max_gap)
  n_episodes = len(episodes)
  if n_episodes < 3:
    raise ValueError('Для temporal_episode_split нужно хотя бы 3 positive-эпизода')

  train_ep_end = max(1, int(np.floor(n_episodes * train_episode_size)))
  val_ep_end = max(train_ep_end + 1, int(np.floor(n_episodes * (train_episode_size + val_episode_size))))
  val_ep_end = min(val_ep_end, n_episodes - 1)

  train_end = episodes[train_ep_end][0]
  val_end = episodes[val_ep_end][0]

  val_start = min(train_end + purge_gap, len(ordered))
  test_start = min(val_end + purge_gap, len(ordered))

  train_df = ordered.iloc[:train_end].reset_index(drop=True)
  val_df = ordered.iloc[val_start:val_end].reset_index(drop=True)
  test_df = ordered.iloc[test_start:].reset_index(drop=True)

  if verbose:
    counts = [int(ordered[target_col].iloc[a:b + 1].sum()) for a, b in episodes]
    print(f"positive episodes: {n_episodes}, event_max_gap={event_max_gap}, positive counts={counts}")
    _print_temporal_split_stats(ordered, train_df, val_df, test_df, target_col, time_col, purge_gap)

  return SplitResult(train=train_df, val=val_df, test=test_df)


def make_supervised_timeseries_frame(
  df: pd.DataFrame,
  target_col: str = 'target_raw',
  time_col: str = 'timestamp',
  target_threshold: float = 55.0,
  forecast_horizon: int = 10,
  lags: tuple[int, ...] = (1, 5, 20),
  rolling_windows: tuple[int, ...] = (5, 20),
  add_current: bool = True,
) -> pd.DataFrame:
  """
  Делает supervised-таблицу для прогноза временного ряда без look-ahead leakage.

  Признаки строятся из текущих и прошлых значений технологических каналов.
  target=1 означает, что через forecast_horizon строк значение target_raw будет
  выше target_threshold.
  """
  ordered = df.sort_values(time_col).reset_index(drop=True).copy()
  feature_cols = [c for c in ordered.columns if c != time_col]

  out = ordered[[time_col]].copy()
  if add_current:
    for col in feature_cols:
      out[col] = ordered[col]

  for lag in lags:
    shifted = ordered[feature_cols].shift(lag)
    shifted.columns = [f'{c}_lag{lag}' for c in feature_cols]
    out = pd.concat([out, shifted], axis=1)

  for window in rolling_windows:
    past = ordered[feature_cols].shift(1)

    roll_mean = past.rolling(window=window, min_periods=window).mean()
    roll_mean.columns = [f'{c}_roll{window}_mean' for c in feature_cols]
    out = pd.concat([out, roll_mean], axis=1)

    roll_std = past.rolling(window=window, min_periods=window).std().replace(0, 0.0)
    roll_std.columns = [f'{c}_roll{window}_std' for c in feature_cols]
    out = pd.concat([out, roll_std], axis=1)

  for lag in lags:
    diff = ordered[feature_cols] - ordered[feature_cols].shift(lag)
    diff.columns = [f'{c}_diff{lag}' for c in feature_cols]
    out = pd.concat([out, diff], axis=1)

  future_target = ordered[target_col].shift(-forecast_horizon)
  out['target'] = (future_target > target_threshold).astype(float)
  out = out.dropna().reset_index(drop=True)
  out['target'] = out['target'].astype(int)

  print(
    f'time-series frame: rows={len(out):,}, features={len(out.columns) - 2:,}, '
    f'target>{target_threshold} at horizon={forecast_horizon}, '
    f'positives={int(out["target"].sum()):,} ({out["target"].mean():.3%})'
  )
  return out
