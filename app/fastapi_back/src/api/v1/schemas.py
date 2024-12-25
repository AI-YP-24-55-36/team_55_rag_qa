from typing import Annotated, Dict, List, Any, TypedDict
from typing_extensions import TypedDict
from pydantic import BaseModel, RootModel

class DatasetEntry(TypedDict):
    context: str
    question: str
    answer: str


DatasetItm = Annotated[DatasetEntry, "Датасет содержащий контекст, вопрос и ответ"]

'''Модель данных для загрузки датасета'''
class DatasetRequest(BaseModel):
    datasets: Dict[str, List[DatasetEntry]]


'''Модель данных для обучения модели'''
class FitRequest(BaseModel):
    hyperparameters: Dict[str, Any]
    model_id: str
    ml_model_type: str
    dataset_nm: str

class FitRequestList(RootModel):
    root: List[FitRequest]


'''Модель данных для загрузки модели'''
class LoadRequest(BaseModel):
    model_id: str


'''Модель данных для предсказания используя предоставленные данные'''
class PredictRequest(BaseModel):
    model_id: str
    question: str

'''Модель данных для вывода списка датасетов.'''
class DsListResponse(BaseModel):
    datasets_nm: List[str]


class ModelElementResponse(BaseModel):
    model_id: str
    type: str

class VectorizeRequest(BaseModel):
    texts: List[str]

'''Модель данных для вывода списка моделей.'''
class ModelListResponse(BaseModel):
    models: List[ModelElementResponse]


class ModelsListResponse(RootModel):
    root: List[ModelListResponse]


'''Модель данных для ответа 1 строки.'''
class ApiResponse(BaseModel):
    message: str


class MessageItem(BaseModel):
    message: str

'''Модель данных для ответа нескольких элементов.'''
class MutlipleApiResponse(RootModel):
    root: List[MessageItem]


'''Модель данных для ответа модели.'''
class FindCntxtResponse(BaseModel):
    context: str
    score: float
    point_id: int
    model_id: str


class FindCntxtsResponse(RootModel):
    root: List[FindCntxtResponse]