import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class FocalLoss(nn.Module):
  """
  Focal Loss для бинарной классификации.
  Подавляет лёгкие примеры и фокусируется на трудных.
 
  Parameters
  ----------
  alpha : float
      Вес положительного класса (дефект). alpha > 0.5 усиливает штраф
      за пропущенные дефекты (recall ↑).
  gamma : float
      Параметр фокусировки. gamma=0 → обычный BCE. gamma=2 стандартно.
  """
  def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma

  def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    probs_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
    focal_weight = alpha_t * (1.0 - probs_true) ** self.gamma

    ce = F.cross_entropy(logits, targets, reduction='none')
    loss = focal_weight * ce
    return loss.mean()
