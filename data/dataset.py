from typing import TypedDict
from torch._prims_common import DeviceLikeType
from torch_geometric.data import Dataset, Data
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
        self.step = step

        self.df['target'] = (self.df['target'] > limit).astype(int)

        self.window_indices = [
            i for i in range(0, len(self.df) - window_size, step)
        ]

        self.device = device

        self.edge_index = self.__build_edge_index()

    def len(self) -> int:
        return len(self.window_indices)

    def get(self, idx) -> Data:
        start = self.window_indices[idx]
        end = start + self.window_size

        window = self.df.iloc[start:end]
        target = self.df.iloc[end]['target']

        x = self.__window_to_node(window)

        y = torch.tensor([target], dtype=torch.long)

        return Data(x=x, edge_index=self.edge_index, y=y)

    def __build_edge_index(self):
        edges = [] 

        for i in range(len(self.active_stages) - 1):
            edges.append([i, i+1])

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __window_to_node(self, window: DataFrame):
        node_features = []

        for name in self.active_stages:
            cols = self.stages[name]
            
            features = window[cols].values.flatten()

            node_features.append(features)

        x = torch.tensor(np.vstack(node_features), dtype=torch.float)

        return x
