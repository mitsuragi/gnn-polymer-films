from dataclasses import dataclass
from io import BytesIO

@dataclass 
class InferenceConfig:
    """
    Parameters
    ----------
    checkpoint_path : str | Path
        Путь к файлу с весами модели (best_model.pt).
    batch_size : int
        Размер батча при прогоне через модель.
    device : str
        "auto" | "cpu" | "cuda" | "mps"
    defect_threshold : float
        Порог вероятности для присвоения класса 1 (дефект).
        По умолчанию 0.5. Уменьшить — повысить recall (меньше пропусков),
        увеличить — повысить precision (меньше ложных тревог).
    target_col : str | None
        Если в DataFrame есть колонка с истинными метками —
        будет посчитаны метрики качества. None — пропустить.
 
    # Параметры датасета — должны совпадать с теми, что использовались при обучении
    edge_strategy : str | list[str]
    edge_threshold : float
    self_loops : bool
    normalize_features : bool
    """
    state_dict_blob: BytesIO
    batch_size: int = 64
    device: str = 'auto'
    defect_threshold: float = 0.5
    target_col: str | None = None 

    edge_strategy: str | list[str] = 'pearson'
    edge_threshold: float = 0.3 
    self_loops: bool = True 
    normalize_features: bool = True
