import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
  GATv2Conv,
  global_add_pool,
  global_mean_pool,
  global_max_pool,
)
from torch_geometric.data import Batch, Data

class GATBlock(nn.Module):
  """
  Один слой графового внимания с нормализацией и остаточным соединением.
 
  Parameters
  ----------
  in_channels : int
  out_channels : int      размерность выхода на голову (head)
  heads : int             число голов внимания
  dropout : float         вероятность dropout на веса внимания и активации
  edge_dim : int | None   размерность edge_attr; None — не используется
  residual : bool         добавлять ли skip-connection
  """
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    heads: int = 4,
    dropout: float = 0.3,
    edge_dim: int | None = 1,
    residual: bool = True,
  ):
    super().__init__()

    self.conv = GATv2Conv(
      in_channels   = in_channels,
      out_channels  = out_channels,
      heads         = heads,
      dropout       = dropout,
      edge_dim      = edge_dim,
      concat        = True,
      add_self_loops= False,
    )

    out_total = out_channels * heads
    self.norm = nn.BatchNorm1d(out_total)
    self.dropout = nn.Dropout(p=dropout)

    self.residual = residual
    if residual and in_channels != out_total:
      self.skip = nn.Linear(in_channels, out_total, bias=False)
    else:
      self.skip = None

  def forward(
    self,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
  ) -> Tensor:
    out = self.conv(x, edge_index, edge_attr=edge_attr)
    out = self.norm(out)
    out = F.elu(out)
    out = self.dropout(out)

    if self.residual:
      res = self.skip(x) if self.skip is not None else x
      out = out + res

    return out


class PolymerGAT(nn.Module):
  """
  Графовая сеть внимания для бинарной классификации дефектов плёнки.
 
  Схема прохождения данных:
      x  →  InputProjection  →  GATBlock × n_layers
         →  GlobalPooling (mean + max + add)
         →  BatchNorm  →  Linear  →  Dropout  →  Linear(2)
 
  Parameters
  ----------
  in_channels : int
      Размерность признака узла из датасета (обычно 1).
  hidden_channels : int
      Число скрытых каналов на голову в каждом GAT-слое.
  n_layers : int
      Число GATBlock-слоёв (рекомендуется 2–4).
  heads : int
      Число голов внимания.
  dropout : float
      Dropout в GAT-блоках и классификационной голове.
  edge_dim : int | None
      Размерность edge_attr. Передать None, чтобы игнорировать веса рёбер.
  pooling : str
      Стратегия глобального пулинга: "mean", "max", "add", "concat".
      "concat" объединяет mean + max + add → более богатое представление.
  """
  def __init__(
    self,
    in_channels: int = 1,
    hidden_channels: int = 32,
    n_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.3,
    edge_dim: int | None = 1,
    pooling: str = 'concat',
  ):
    super().__init__()

    if n_layers < 1:
      raise ValueError('n_layers должно быть >= 1')

    self.pooling = pooling

    self.input_proj = nn.Sequential(
      nn.Linear(in_channels, hidden_channels),
      nn.BatchNorm1d(hidden_channels),
      nn.ELU(),
    )

    self.gat_blocks = nn.ModuleList()
    in_ch = hidden_channels
    for i in range(n_layers):
      is_last = (i == n_layers - 1)
      block = GATBlock(
        in_channels=in_ch,
        out_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        edge_dim=edge_dim,
        residual=True,
      )
      self.gat_blocks.append(block)
      in_ch = hidden_channels * heads

    graph_embed_dim = in_ch * 3 if pooling == 'concat' else in_ch

    self.classifier = nn.Sequential(
      nn.BatchNorm1d(graph_embed_dim),
      nn.Linear(graph_embed_dim, hidden_channels*2),
      nn.ELU(),
      nn.Dropout(p=dropout),
      nn.Linear(hidden_channels*2, 2),
    )

  def forward(self, data: Data | Batch) -> Tensor:
    """
    Parameters
    ----------
    data : torch_geometric.data.Data | Batch
        Граф или батч графов из DataLoader.
 
    Returns
    -------
    Tensor  shape (B, 2) — логиты классов
    """
    x, edge_index, edge_attr, batch = (
      data.x,
      data.edge_index,
      data.edge_attr,
      data.batch,
    )

    x = self.input_proj(x)

    for block in self.gat_blocks:
      x = block(x, edge_index, edge_attr)

    if self.pooling == 'concat':
      graph_emb = torch.cat([
        global_mean_pool(x, batch),
        global_max_pool(x, batch),
        global_add_pool(x, batch),
      ], dim=-1)
    elif self.pooling == 'mean':
      graph_emb = global_mean_pool(x, batch)
    elif self.pooling == 'max':
      graph_emb = global_max_pool(x, batch)
    elif self.pooling == 'add':
      graph_emb = global_add_pool(x, batch)
    else:
      raise ValueError(f'Неизвестый pooling: {self.pooling!r}')

    logits = self.classifier(graph_emb)

    return logits

  @torch.no_grad()
  def predict_proba(self, data: Data | Batch) -> Tensor:
    """Возвращает вероятности классов (B, 2)."""
    self.eval()
    return F.softmax(self(data), dim=-1)

  @torch.no_grad()
  def predict(self, data: Data | Batch) -> Tensor:
    """Возвращает предсказанный класс (B,)."""
    return self.predict_proba(data).argmax(dim=-1)

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(
  in_channels: int = 1,
  hidden_channels: int = 32,
  n_layers: int = 3,
  heads: int = 4,
  dropout: float = 0.3,
  edge_dim: int | None = 1,
  pooling: str = 'concat',
) -> PolymerGAT:
  """Создаёт и возвращает инициализированную модель."""
  model = PolymerGAT(
    in_channels     =in_channels,
    hidden_channels =hidden_channels,
    n_layers        =n_layers,
    heads           =heads,
    dropout         =dropout,
    edge_dim        =edge_dim,
    pooling         = pooling,
  )

  return model
