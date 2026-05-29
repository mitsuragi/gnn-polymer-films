import time
from pathlib import Path
import numpy as np 
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn

from .metrics_logger import EpochMetrics, MetricsLogger
from .model import PolymerGAT
from .early_stopping import EarlyStopping
from .train_config import TrainConfig
from .focal_loss import FocalLoss

def compute_metrics(
  epoch: int,
  phase: str,
  loss: float,
  y_true: np.ndarray,      # (N,)  int  0/1
  y_prob: np.ndarray,      # (N,)  float  P(class=1)
  y_pred: np.ndarray,      # (N,)  int  0/1
  duration_sec: float,
) -> EpochMetrics:
  """
  Считает все метрики по собранным за эпоху предсказаниям.
 
  Регрессионные метрики применяются к вероятностям y_prob vs y_true,
  что позволяет оценить качество калибровки модели.
  """
 
  # ── Регрессионные метрики (вероятность vs бинарная метка) ────────────────
  errors   = y_prob - y_true                          # (N,)
  mae      = float(np.mean(np.abs(errors)))
  rmse     = float(np.sqrt(np.mean(errors ** 2)))
 
  # WAPE: взвешенная абсолютная процентная ошибка
  # знаменатель — сумма истинных меток; если все 0 → wape=0 (деление защищено)
  denom = np.sum(np.abs(y_true))
  wape  = float(np.sum(np.abs(errors)) / denom) if denom > 0 else 0.0
 
  # ── Классификационные метрики (жёсткие предсказания) ────────────────────
  tp = int(np.sum((y_pred == 1) & (y_true == 1)))
  fp = int(np.sum((y_pred == 1) & (y_true == 0)))
  fn = int(np.sum((y_pred == 0) & (y_true == 1)))
  tn = int(np.sum((y_pred == 0) & (y_true == 0)))
 
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1        = (
      2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0.0
  )
  accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
 
  return EpochMetrics(
    epoch        = epoch,
    phase        = phase,
    loss         = loss,
    mae          = mae,
    rmse         = rmse,
    wape         = wape,
    precision    = float(precision),
    recall       = float(recall),
    f1           = float(f1),
    accuracy     = float(accuracy),
    duration_sec = duration_sec,
  )

def _run_epoch(
  model: PolymerGAT,
  loader: DataLoader,
  criterion: nn.Module,
  device: torch.device,
  optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """
  Выполняет один проход по DataLoader.
 
  Returns
  -------
  avg_loss, y_true, y_prob, y_pred
  """
  is_train = optimizer is not None
  model.train(is_train)
  ctx = torch.enable_grad() if is_train else torch.no_grad()

  total_loss = 0.0
  all_true, all_prob, all_pred = [], [], []

  with ctx:
    for batch in loader:
      batch = batch.to(device)

      logits = model(batch)

      loss = criterion(logits, batch.y)

      if is_train:
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping — стабилизирует обучение на графах
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

      probs = torch.softmax(logits, dim=-1)[:, 1] # P(дефект)
      preds = logits.argmax(dim=-1)

      total_loss += loss.item() * batch.num_graphs
      all_true.append(batch.y.cpu().numpy())
      all_prob.append(probs.detach().cpu().numpy())
      all_pred.append(preds.detach().cpu().numpy())

  n = sum(len(t) for t in all_true)
  avg_loss = total_loss / n if n > 0 else 0.0
  return (
    avg_loss,
    np.concatenate(all_true),
    np.concatenate(all_prob),
    np.concatenate(all_pred),
  )

def train(
  model: PolymerGAT,
  train_loader: DataLoader,
  val_loader: DataLoader,
  test_loader: DataLoader | None = None,
  config: TrainConfig | None = None,
) -> MetricsLogger:
  """
  Полный цикл обучения: train → val (каждую эпоху) → test (в конце).
 
  Parameters
  ----------
  model         : инициализированная PolymerGAT
  train_loader  : DataLoader обучающего датасета
  val_loader    : DataLoader валидационного датасета
  test_loader   : DataLoader тестового датасета (опционально)
  config        : TrainConfig с гиперпараметрами
 
  Returns
  -------
  MetricsLogger со всей историей метрик
  """
  if config is None:
    config = TrainConfig()

  if config.device == 'auto':
    device = torch.device(
      'cuda' if torch.cuda.is_available()
      else 'mps' if torch.backends.mps.is_available()
      else 'cpu'
    )
  else:
    device = torch.device(config.device)

  model = model.to(device)
  print(f'Устройство: {device}')
  print(f"Параметров модели: {model.count_parameters():,}\n")

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = config.learning_rate,
    weight_decay = config.weight_decay,
  )

  criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = config.es_mode,
    factor = config.lr_factor,
    patience = config.lr_patience,
    min_lr = config.lr_min,
  )

  early_stopping = EarlyStopping(
    patience = config.es_patience,
    min_delta = config.es_min_delta,
    mode = config.es_mode,
    checkpoint_path = config.checkpoint_path,
  )
  logger = MetricsLogger(log_path=config.log_csv_path)

  for epoch in range(1, config.n_epochs + 1):
    # Train
    t0 = time.perf_counter()
    tr_loss, tr_true, tr_prob, tr_pred = _run_epoch(
      model, train_loader, criterion, device, optimizer
    )
    train_metrics = compute_metrics(
      epoch, 'train', tr_loss,
      tr_true, tr_prob, tr_pred,
      duration_sec=time.perf_counter() - t0,
    )
    logger.log(train_metrics)

    # Val
    t0 = time.perf_counter()
    val_loss, val_true, val_prob, val_pred = _run_epoch(
      model, val_loader, criterion, device, optimizer=None
    )
    val_metrics = compute_metrics(
      epoch, 'val', val_loss,
      val_true, val_prob, val_pred,
      duration_sec=time.perf_counter() - t0,
    )
    logger.log(val_metrics)

    # Scheduler step
    monitor_value = getattr(val_metrics, config.es_monitor)
    scheduler.step(monitor_value)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"lr={current_lr:.2e}  "
          f"es_counter={early_stopping.epochs_without_improvement}/{config.es_patience}")

    # Early stopping
    should_continue = early_stopping.step(monitor_value, model)
    if not should_continue:
      print(f"\n⏹  Early stopping на эпохе {epoch}. "
            f"Лучший val {config.es_monitor}={early_stopping.best_score:.4f}")
      break

  best_ckpt = Path(config.checkpoint_path)
  if best_ckpt.exists():
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f'Загружены лучшие веса из {best_ckpt}')

  if test_loader is not None:
    t0 = time.perf_counter()
    test_loss, test_true, test_prob, test_pred = _run_epoch(
      model, test_loader, criterion, device, optimizer=None
    )
    test_metrics = compute_metrics(
      epoch=0, phase='test',
      loss=test_loss,
      y_true=test_true, y_prob=test_prob, y_pred=test_pred,
      duration_sec=time.perf_counter() - t0,
    )
    logger.log(test_metrics)
    print(f"\n{'='*70}")
    print('Итоговые метрики на тесте')
    print(f"{'='*70}")
    print(test_metrics)

  return logger
