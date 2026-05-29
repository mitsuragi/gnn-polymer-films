from dataclasses import dataclass, field 
import warnings
from pathlib import Path
from typing import Optional
import io

import numpy as np 
import pandas as pd 
import torch 
import torch.nn.functional as F 
from torch_geometric.loader import DataLoader

from data import PolyFilmDataset 
from gnn import PolymerGAT, build_model

from .inference_config import InferenceConfig

@dataclass 
class PredictionResult:
    """
    Атрибуты
    ---------
    predictions : np.ndarray  (N,)  int    — 0 / 1 для каждой строки
    probabilities : np.ndarray (N,) float  — P(дефект) ∈ [0, 1]
    defect_threshold : float               — использованный порог
    summary : pd.DataFrame                 — удобная таблица для анализа
    metrics : dict | None                  — метрики, если были истинные метки
    """
    predictions: np.ndarray
    probabilities: np.ndarray
    defect_threshold: float 
    summary: pd.DataFrame
    metrics: dict | None = None

    def __repr__(self) -> str:
        n = len(self.predictions)
        n_def = int(self.predictions.sum())
        pct = 100 * n_def / n if n else 0
        lines = [
            f'PredictionResult(n={n})',
            f"  Дефектов: {n_def} / {n} ({pct:.1f})",
            f"  Порог: {self.defect_threshold}"
        ]

        if self.metrics:
            m = self.metrics
            lines.append(
                f"  Precision:   {m.get('precision', 0):.3f}  "
                f"Recall: {m.get('recall', 0):.3f}  "
                f"F1: {m.get('f1', 0):.3f}"
            )
        return "\n".join(lines)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
 
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / len(y_true) if len(y_true) else 0.0
 
    errors = y_prob - y_true
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    denom  = np.sum(np.abs(y_true))
    wape   = float(np.sum(np.abs(errors)) / denom) if denom > 0 else 0.0
 
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        mae=mae, rmse=rmse, wape=wape,
    )

class DefectPredictor:
    """
    Обёртка над обученной GAT-моделью для прогнозирования дефектов.
 
    Использование
    -------------
    predictor = DefectPredictor(config)
    result    = predictor.predict(df)
 
    print(result)
    print(result.summary)      # строка за строкой
    print(result.metrics)      # если передан target_col
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._device = self._resolve_device(config.device)
        self._model: PolymerGAT | None = None

    def load_model(
        self,
        model: PolymerGAT | None = None,
        model_kwargs: dict | None = None,
    ) -> "DefectPredictor":

        """
        Загружает веса модели.
 
        Parameters
        ----------
        model : PolymerGAT | None
            Готовый экземпляр модели (архитектура уже создана).
            Если None — модель будет создана через build_model(**model_kwargs).
        model_kwargs : dict | None
            Аргументы для build_model, если model=None.
            Пример: {"in_channels":1, "hidden_channels":32, "n_layers":3, ...}
        """
        if model is not None:
            self._model = model
        else:
            kwargs = model_kwargs or {}
            self._model = build_model(**kwargs)

        buffer = io.BytesIO(self.config.state_dict_blob)
        state = torch.load(buffer, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

        print(f'Модель загружена | устройство: {self._device}') 
        return self

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Делает предсказания для всех строк DataFrame.
 
        Parameters
        ----------
        df : pd.DataFrame
            Строки — наблюдения (временные шаги / партии).
            Столбцы — показания датчиков + опционально target_col.
 
        Returns
        -------
        PredictionResult
        """
        if self._model is None:
            raise RuntimeError('Сначала вызовите load_model()')

        cfg = self.config

        y_true: np.ndarray | None = None
        if cfg.target_col and cfg.target_col in df.columns:
            y_true = df[cfg.target_col].values.astype(int)
            feature_df = df.drop(columns=[cfg.target_col])
        else:
            feature_df = df.copy()

        if feature_df.shape[0] == 0:
            raise ValueError('DataFrame пустой')

        inference_df = feature_df.copy()
        _FAKE_TARGET = '__target__'
        inference_df[_FAKE_TARGET] = 0

        dataset = PolyFilmDataset(
            inference_df,
            target_col=_FAKE_TARGET,
            edge_strategy=cfg.edge_strategy,
            threshold=cfg.edge_threshold,
            self_loops=cfg.self_loops,
            normalize_features=cfg.normalize_features
        )
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

        all_probs, all_preds = self._run_inference(loader, cfg.defect_threshold)

        summary = self._build_summary(feature_df, all_probs, all_preds, y_true)

        # Метрики
        metrics = None
        if y_true is not None:
            metrics = _compute_metrics(y_true, all_preds, all_probs)
            self._print_metrics(metrics)

        return PredictionResult(
            predictions=all_preds,
            probabilities=all_probs,
            defect_threshold=cfg.defect_threshold,
            summary=summary,
            metrics=metrics,
        )
    
    def predict_single(self, row: pd.Series | dict,
                       feature_names: list[str]) -> dict:
        """
        Предсказание для одной строки (одного момента времени).
 
        Parameters
        ----------
        row : pd.Series | dict   значения датчиков
        feature_names : list[str]  имена признаков в нужном порядке
 
        Returns
        -------
        dict с ключами: prediction, probability, risk_level
        """
        if isinstance(row, dict):
            row = pd.Series(row)

        df_single = pd.DataFrame([row[feature_names]])
        result = self.predict(df_single)
        prob = float(result.probabilities[0])
        pred = int(result.predictions[0])

        return {
            'prediction': pred,
            'probability': prob,
        }

    @torch.no_grad()
    def _run_inference(
        self,
        loader: DataLoader,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_probs = []
        for batch in loader:
            batch = batch.to(self._device)
            logits = self._model(batch)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())

        probs_arr = np.concatenate(all_probs)
        preds_arr = (probs_arr >= threshold).astype(int)
        return probs_arr, preds_arr

    @staticmethod
    def _build_summary(
        feature_df: pd.DataFrame,
        probs: np.ndarray,
        preds: np.ndarray,
        y_true: np.ndarray | None,
    ) -> pd.DataFrame:
        summary = feature_df.copy().reset_index(drop=True)
        summary['prob_defect'] = probs.round(4)
        summary['prediction'] = preds

        if y_true is not None:
            summary['true_label'] = y_true 
            summary['correct'] = (preds == y_true)
        
        return summary

    @staticmethod
    def _print_metrics(m: dict) -> None:
        sep = "─" * 52
        print(f"\n{sep}")
        print("МЕТРИКИ КАЧЕСТВА ПРОГНОЗА")
        print(sep)
        print(f"  Accuracy   : {m['accuracy']:.4f}")
        print(f"  Precision  : {m['precision']:.4f}")
        print(f"  Recall     : {m['recall']:.4f}")
        print(f"  F1         : {m['f1']:.4f}")
        print(f"  MAE        : {m['mae']:.4f}")
        print(f"  RMSE       : {m['rmse']:.4f}")
        print(f"  WAPE       : {m['wape']:.4f}")
        print(f"  TP/FP/FN/TN: {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}")
        print(sep)

    @staticmethod 
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available(): return torch.device('cuda')
            if torch.backends.mps.is_available(): return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device)
