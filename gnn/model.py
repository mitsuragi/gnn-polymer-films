import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
  GATv2Conv,
  GraphNorm,
  global_add_pool,
  global_mean_pool,
  global_max_pool,
)
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax

try:
    from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention
except ImportError:
    from torch_geometric.nn import GlobalAttention

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
    self.norm = GraphNorm(out_total)
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
    batch: Tensor | None = None,
  ) -> Tensor:
    out = self.conv(x, edge_index, edge_attr=edge_attr)
    out = self.norm(out, batch)
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

    POOLINGS = ('attention', 'concat', 'mean', 'max', 'add')

    def __init__(
        self,
        n_nodes: int,
        node_emb_dim: int = 16,
        in_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int | None = 1,
        pooling: str = 'attention',
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError('n_layers должно быть >= 1')

        if pooling not in self.POOLINGS:
            raise ValueError(f'pooling должен быть одним из {self.POOLINGS}, получено {pooling}')

        self.n_nodes = n_nodes
        self.pooling = pooling

        self.node_emb = nn.Embedding(n_nodes, node_emb_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + node_emb_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
        )

        self.gat_blocks = nn.ModuleList()
        in_ch = hidden_channels
        for _ in range(n_layers):
            self.gat_blocks.append(GATBlock(
                in_channels=in_ch,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                residual=True,
            ))
            in_ch = hidden_channels * heads

        self._node_dim = in_ch

        if pooling == 'attention':
            self.att_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(in_ch, in_ch // 2),
                    nn.ELU(),
                    nn.Linear(in_ch // 2, 1)
                )
            )
            graph_embed_dim = in_ch
        elif pooling == 'concat':
            graph_embed_dim = in_ch * 3 
        else:
            graph_embed_dim = in_ch

        self.classifier = nn.Sequential(
            nn.LayerNorm(graph_embed_dim),
            nn.Linear(graph_embed_dim, hidden_channels*2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels*2, 1),
        )

    def encode(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        n_graphs = int(batch.max().item()) + 1 
        node_ids = torch.arange(self.n_nodes, device=x.device).repeat(n_graphs)

        x = torch.cat([x, self.node_emb(node_ids)], dim=-1)
        x = self.input_proj(x)

        for block in self.gat_blocks:
            x = block(x, edge_index, edge_attr, batch)

        return x, batch

    def forward(self, data: Data | Batch) -> Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data | Batch
            Граф или батч графов из DataLoader.
 
        Returns
        -------
        Tensor  shape (B, 1) — логиты классов
        """
        x, batch = self.encode(data)

        if self.pooling == 'attention':
            graph_emb = self.att_pool(x, batch)
        elif self.pooling == 'concat':
            graph_emb = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ], dim=-1)
        elif self.pooling == 'mean':
            graph_emb = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(x, batch)
        else:
            graph_emb = global_add_pool(x, batch)

        return self.classifier(graph_emb)

    @torch.no_grad()
    def predict_proba(self, data: Data | Batch) -> Tensor:
        """Вероятности позитивного класса (B,)."""
        self.eval()
        return torch.sigmoid(self(data)).squeeze(1)

    @torch.no_grad()
    def predict(self, data: Data | Batch, threshold: float = 0.5) -> Tensor:
        """Возвращает предсказанный класс (B,)."""
        return (self.predict_proba(data) >= threshold).long()

    @torch.no_grad()
    def node_attention(self, data: Data | Batch) -> Tensor:
        if self.pooling != 'attention':
            raise RuntimeError("node_attention доступен только при pooling='attention'")
        self.eval()
        x, batch = self.encode(data)
        gate = self.att_pool.gate_nn(x).squeeze(-1)
        return softmax(gate, batch)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(
    n_nodes: int,
    node_emb_dim: int = 16,
    in_channels: int = 1,
    hidden_channels: int = 32,
    n_layers: int = 2,
    heads: int = 2,
    dropout: float = 0.4,
    edge_dim: int | None = 1,
    pooling: str = 'attention',
) -> PolymerGAT:
    return PolymerGAT(
        n_nodes         = n_nodes,
        node_emb_dim    = node_emb_dim,
        in_channels     = in_channels,
        hidden_channels = hidden_channels,
        n_layers        = n_layers,
        heads           = heads,
        dropout         = dropout,
        edge_dim        = edge_dim,
        pooling         = pooling,
    )
