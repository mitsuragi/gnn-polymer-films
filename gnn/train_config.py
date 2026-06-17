from dataclasses import dataclass

@dataclass
class TrainConfig:
  n_epochs: int = 100
  learning_rate: float = 1e-3
  weight_decay: float = 1e-4
  batch_size: int = 32
  num_workers: int = 0

  pos_weight: float | None = None

  use_sampler: bool = True
  sampler_kind: str = 'event_aware'
  sampler_pos_ratio: float = 0.3
  sampler_context_radius: int = 20
  sampler_hard_neg_ratio: float = 0.50

  # Порог: только для информативных prec/rec/f1 в логах во время обучения.
  # На обучение и выбор модели он больше НЕ влияет.
  defect_threshold: float = 0.5

  # Финальный подбор порога (один раз, по лучшему чекпойнту)
  threshold_metric: str = 'f1'
  threshold_min: float = 0.01
  threshold_max: float = 0.90
  threshold_steps: int = 100

  # Early stopping / scheduler — на PR-AUC
  es_monitor: str = 'pr_auc'
  es_mode: str = 'max'
  es_patience: int = 25
  es_min_delta: float = 1e-4

  lr_factor: float = 0.5
  lr_patience: int = 8
  lr_min: float = 1e-6

  device: str = 'auto'
  checkpoint_path: str = 'best_model.pt'
  log_csv_path: str | None = None
