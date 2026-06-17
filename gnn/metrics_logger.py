import csv
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class EpochMetrics:
  epoch:     int
  phase:     str          # "train" | "val" | "test"
  loss:      float
 
  # Классификационные метрики (на жёстких предсказаниях)
  precision: float = 0.0
  recall:    float = 0.0
  f1:        float = 0.0
  accuracy:  float = 0.0
  pr_auc:    float = 0.0
 
  duration_sec: float = 0.0
 
  def __str__(self) -> str:
    return (
        f"[{self.phase.upper():>5}] epoch={self.epoch:>3d} | "
        f"loss={self.loss:.4f}  pr_auc={self.pr_auc:.4f} "
        f"prec={self.precision:.4f}  rec={self.recall:.4f}  "
        f"f1={self.f1:.4f}  acc={self.accuracy:.4f}  "
        f"({self.duration_sec:.1f}s)"
    )

class MetricsLogger:
  def __init__(self, log_path: str | Path | None = None):
    self.log_path = Path(log_path) if log_path else None
    if self.log_path is not None:
      with self.log_path.open('w') as f:
        pass

    self._csv_initialized = False
    self.history: list[EpochMetrics] = []

  def log(self, metrics: EpochMetrics):
    self.history.append(metrics)
    print(metrics)

    if self.log_path is not None:
      row = asdict(metrics)
      write_header = not self._csv_initialized and not self.log_path.exists()
      with self.log_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
          writer.writeheader()
        writer.writerow(row)
      self._csv_initialized = True

  def best(self, phase: str = 'val', key: str = 'loss', mode: str = 'min') -> EpochMetrics:
    subset = [m for m in self.history if m.phase == phase]
    if not subset:
      raise ValueError(f"Нет записей для фазы '{phase}'")
    return min(subset, key=lambda m: getattr(m, key)) if mode == 'min' \
    else max(subset, key=lambda m: getattr(m, key))
