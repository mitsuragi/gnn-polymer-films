from torch._prims_common import DeviceLikeType
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from pandas import DataFrame

StagesDict = dict[str, list[str] | None]

class PolyFilmDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        window_size,
        stage_dict: StagesDict,
        step: int = 1,
        limit: int = 0,
        device: DeviceLikeType | None = None
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)

        self.stages = stage_dict
        
        self.active_stages = []
        for name, cols in stage_dict.items():
            if cols is not None:
                self.active_stages.append(name)

        self.window_size = window_size

        self.max_features = max(
            len(cols) for cols in stage_dict.values() if cols is not None
        ) * self.window_size

        self.step = step

        # self.df['target'] = (self.df['target'] > limit).astype(int)

        self.window_indices = [
            i for i in range(0, len(self.df) - self.window_size, step)
        ]

        self.device = device

        self.edge_index = self.build_edge_index()

    def len(self) -> int:
        return len(self.window_indices)

    def get(self, idx) -> Data:
        start = self.window_indices[idx]
        end = start + self.window_size

        window = self.df.iloc[start:end]
        target = self.df.iloc[end]['target']

        x = self.window_to_node(window)

        y = torch.tensor([target], dtype=torch.long)

        return Data(x=x, edge_index=self.edge_index, y=y)

    def build_edge_index(self):
        edges = [] 

        for i in range(len(self.active_stages) - 1):
            edges.append([i, i+1])

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def window_to_node(self, window: DataFrame):
        node_features = []

        for name in self.active_stages:
            cols = self.stages[name]
            
            features = window[cols].values.flatten()

            if len(features) < self.max_features:
                pad = self.max_features - len(features)
                features = np.pad(features, (0, pad))

            node_features.append(features)

        x = torch.tensor(np.vstack(node_features), dtype=torch.float)

        return x

def get_datasets(
    df,
    window_size,
    stage_dict,
    step,
    limit
):
    train_size = int(len(df) * 0.7)
    val_size = int((len(df) - train_size) / 2)

    df['target'] = (df['target'] > 0).astype(int)

    train_data = df.iloc[:train_size]
    eval_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    pos = (train_data['target'] == 0).sum()
    neg = (train_data['target'] == 1).sum()

    print(pos)
    print(neg)

    pos_weight = neg / pos

    train_ds = PolyFilmDataset(
        train_data,
        window_size,
        stage_dict,
        step,
        limit
    )

    eval_ds = PolyFilmDataset(
        eval_data,
        window_size,
        stage_dict,
        step,
        limit
    )

    test_ds = PolyFilmDataset(
        test_data,
        window_size,
        stage_dict,
        step,
        limit
    )

    return (train_ds, eval_ds, test_ds, pos_weight)

def get_dataloaders(
    train_ds,
    eval_ds,
    test_ds,
    batch_size
):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return (train_dl, eval_dl, test_dl)
