import time
from pathlib import Path
import numpy as np 
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from data.dataset import PolyFilmDataset
from gnn.sampler import make_event_aware_sampler, make_weighted_sampler

from .metrics_logger import EpochMetrics, MetricsLogger
from .model import PolymerGAT
from .early_stopping import EarlyStopping
from .train_config import TrainConfig

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
 
    # ── Классификационные метрики (жёсткие предсказания) ────────────────────
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy  = float((y_pred == y_true).mean())

    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        pr_auc = 0.0
 
    return EpochMetrics(
        epoch        = epoch,
        phase        = phase,
        loss         = loss,
        precision    = precision,
        recall       = recall,
        f1           = f1,
        accuracy     = accuracy,
        pr_auc       = pr_auc,
        duration_sec = duration_sec,
    )

def _run_epoch(
    model: PolymerGAT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
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

            logits = model(batch).squeeze(1)

            loss = criterion(logits, batch.y.float())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.sigmoid(logits).detach()
            preds = (probs >= threshold).long()

            total_loss += loss.item() * batch.num_graphs
            all_true.append(batch.y.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    n = sum(len(t) for t in all_true)
    avg_loss = total_loss / n if n > 0 else 0.0
    return (
        avg_loss,
        np.concatenate(all_true),
        np.concatenate(all_prob),
        np.concatenate(all_pred),
    )

def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    min_threshold: float = 0.02,
    max_threshold: float = 0.80,
    steps: int = 79,
) -> tuple[float, float]:
    """Подбирает порог на валидации для редкого positive-класса."""
    thresholds = np.linspace(min_threshold, max_threshold, steps)
    best_threshold = float(thresholds[0])
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        m = compute_metrics(
            epoch=0,
            phase='threshold_search',
            loss=0.0,
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            duration_sec=0.0,
        )
        score = getattr(m, metric)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score

def train(
    model: PolymerGAT,
    train_dataset: PolyFilmDataset,
    val_loader: DataLoader,
    test_loader: DataLoader | None = None,
    config: TrainConfig | None = None,
    train_labels: np.ndarray | None = None,
): # -> MetricsLogger:
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

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = model.to(device)
    print(f'Устройство: {device}')
    print(f"Параметров модели: {model.count_parameters():,}\n")

    if config.pos_weight is not None:
        pw = torch.tensor([config.pos_weight], dtype=torch.float32).to(device)
    elif train_labels is not None and not config.use_sampler:
        n_pos = int(train_labels.sum())
        n_neg = len(train_labels) - n_pos
        pw_val = n_neg / max(n_pos, 1)
        pw = torch.tensor([pw_val], dtype=torch.float32).to(device)
        print(f'pos_weight={pw_val:.1f} (n_neg={n_neg}, n_pos={n_pos})')
    else:
        pw = None
        if train_labels is not None and config.use_sampler:
            print('pos_weight отключен: дисбаланс компенсирует WeightedRandomSampler')
        else:
            print('pos_weight не задан')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    if config.use_sampler and train_labels is not None:
        if config.sampler_kind == 'event_aware':
            sampler = make_event_aware_sampler(
                train_labels,
                pos_ratio=config.sampler_pos_ratio,
                context_radius=config.sampler_context_radius,
                hard_neg_ratio=config.sampler_hard_neg_ratio,
            )

            print(
                f"EventAwareSampler: pos_ratio={config.sampler_pos_ratio}, "
                f"context_radius={config.sampler_context_radius}, "
                f"hard_neg_ratio={config.sampler_hard_neg_ratio}"
            )
        elif config.sampler_kind == 'weighted':
            sampler = make_weighted_sampler(train_labels, config.sampler_pos_ratio)
            print(f'WeightedRandomSampler: pos_ratio={config.sampler_pos_ratio}')
        else:
            raise ValueError(f'Неизвестный sampler_kind: {config.sampler_kind}')
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay = config.weight_decay,
    )

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
            model, train_loader, criterion, device, config.defect_threshold, optimizer
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
            model, val_loader, criterion, device, config.defect_threshold, optimizer=None
        )
        adaptive_thr = np.quantile(val_prob, 1.0-val_true.mean())
        val_pred = (val_prob >= adaptive_thr).astype(int)
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

    _, fin_val_true, fin_val_prob, _ = _run_epoch(
        model, val_loader, criterion, device,
        threshold=0.5, optimizer=None
    )
    best_threshold, best_thr_score = _find_best_threshold(
        fin_val_true, fin_val_prob,
        metric=config.threshold_metric,
        min_threshold=config.threshold_min,
        max_threshold=config.threshold_max,
        steps=config.threshold_steps,
    )
    print(f'Финальный threshold по val: {best_threshold:.3f} '
          f'({config.threshold_metric}={best_thr_score:.4f})')

    if test_loader is not None:
        t0 = time.perf_counter()
        test_loss, test_true, test_prob, test_pred = _run_epoch(
            model, test_loader, criterion, device, best_threshold, optimizer=None
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

    return model, best_threshold

def predict(
    model,
    loader,
    threshold
):
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    loss, true, prob, pred = _run_epoch(
        model, loader, criterion, device, threshold, optimizer=None
    )
    metrics = compute_metrics(
        epoch=0, phase='test',
        loss=loss,
        y_true=true, y_prob=prob, y_pred=pred,
        duration_sec=time.perf_counter(),
    )
    print(metrics)

    return true, prob, pred
