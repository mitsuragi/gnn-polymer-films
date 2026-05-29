from dataclasses import dataclass

@dataclass
class TrainConfig:
  # main params
  n_epochs: int = 100
  learning_rate: float = 1e-3
  weight_decay: float = 1e-4

  # early stopping params
  es_patience: int = 10
  es_min_delta: float = 1e-4
  es_monitor: str = 'loss'  # метрика val для early stopping
  es_mode: str = 'min'  # 'min' | 'max'

  # LR scheduler
  lr_factor: float = 0.5
  lr_patience: int = 7
  lr_min: float = 1e-6

  # focal loss
  focal_alpha: float = 0.75
  focal_gamma: float = 2.0

  # paths
  checkpoint_path: str = 'best_model.pt'
  log_csv_path: str = 'training_log.csv'

  # device
  device: str = 'auto' # 'auto' | 'cpu' | 'cuda' | 'mps'
