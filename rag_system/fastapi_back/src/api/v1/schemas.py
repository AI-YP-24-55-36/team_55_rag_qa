from typing import Annotated, Dict, List, Any, TypedDict, Optional
from pydantic import BaseModel, RootModel


class DatasetEntry(TypedDict):
    context: str
    question: str
    answer: str


DatasetItm = Annotated[DatasetEntry,
                       "Датасет содержащий контекст, вопрос и ответ"]


class DatasetRequest(BaseModel):
    """Модель данных для загрузки датасета"""
    datasets: Dict[str, List[DatasetItm]]


class FitRequest(BaseModel):
    """Модель данных для обучения модели"""
    hyperparameters: Dict[str, Any]
    model_id: str
    ml_model_type: str
    dataset_nm: str


class FitRequestList(RootModel):
    root: List[FitRequest]


class LoadRequest(BaseModel):
    """Модель данных для загрузки модели"""
    model_id: str


class PredictRequest(BaseModel):
    """Модель данных для предсказания используя предоставленные данные"""
    model_id: str
    question: str


class DsListResponse(BaseModel):
    """Модель данных для вывода списка датасетов."""
    datasets_nm: List[str]


class ModelElementResponse(BaseModel):
    model_id: str
    type: str
    hyperparameters: Dict[str, Any]


class VectorizeRequest(BaseModel):
    texts: List[str]


class ModelListResponse(BaseModel):
    """Модель данных для вывода списка моделей."""
    models: List[ModelElementResponse]


class ModelsListResponse(RootModel):
    root: List[ModelListResponse]


class ApiResponse(BaseModel):
    """Модель данных для ответа 1 строки."""
    message: str


class MessageItem(BaseModel):
    message: str


class MutlipleApiResponse(RootModel):
    """Модель данных для ответа нескольких элементов."""
    root: List[MessageItem]


class FindCntxtResponse(BaseModel):
    """Модель данных для ответа модели."""
    context: str
    score: float
    point_id: int
    model_id: str


class CheckRequest(BaseModel):
    """Модель данных для тестирования модели"""
    model_id: str
    threshold: Optional[int] = None


class FindCntxtsResponse(RootModel):
    root: List[FindCntxtResponse]


class AccuracyResponse(BaseModel):
    accuracy: float
    timings: List[float]


class FindAnswerResponse(BaseModel):
    question: str
    answer: str
    used_contexts: List[dict]
    model_used: str
