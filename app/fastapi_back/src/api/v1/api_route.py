from http import HTTPStatus
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from sklearn.feature_extraction.text import TfidfVectorizer

from ...qdrant.load_qdrant import save_vectors_batch, search_similar_texts, check_questions
from ...logger import api_logger

from .schemas import (
    MutlipleApiResponse,
    Annotated,
    DatasetRequest,
    FitRequestList,
    LoadRequest,
    FindCntxtsResponse,
    PredictRequest,
    AccuracyResponse,
    CheckRequest,
    DsListResponse,
    ModelsListResponse,
)

DB = {"datasets": {}, "models": {}}

router = APIRouter(prefix="/api/v1/models")
qdrant_client = QdrantClient(url="http://qdrant:6333", timeout=1000)


# API endpoints
@router.post("/load_dataset", response_model=MutlipleApiResponse, status_code=HTTPStatus.CREATED,
             description='Загрузка датасета')
async def fit(request: Annotated[DatasetRequest, 'Датасеты в формате массива списков']):
    """Функция обучения модели"""
    global DB
    rs = []
    # Работа с несколькими датасетами
    if DB['datasets']:
        del DB['datasets']
        del DB['models']
    DB['datasets'] = {}
    DB['models'] = {}
    datasets = request.datasets
    for dataset_nm, dataset in datasets.items():
        if dataset_nm in DB['datasets']:
            api_logger.error("Dataset '%s' download error", dataset_nm)
            raise HTTPException(
                status_code=400, detail="Датасет с таким названием уже был загружен")
        DB['datasets'][dataset_nm] = pd.DataFrame(dataset)
        rs.append({'message': f"Датасет '{dataset_nm}' загружен"})
        api_logger.info("Dataset '%s' downloaded", dataset_nm)

    return MutlipleApiResponse(root=rs)


@router.post("/fit_save", response_model=MutlipleApiResponse, status_code=HTTPStatus.CREATED,
             description='Обучение на датасете и загрузка в qdrant')
async def fit_save(request: Annotated[FitRequestList, '''Список моделей
                                      и гиперпараметров для обучения,
                                      векторизации и сохранения датасета''']):
    """Обучение на датасете и загрузка в qdrant"""
    global DB
    global qdrant_client
    # Обучение нескольких моделей в 1 запросе
    request = request.root[0]
    model_id = request.model_id
    hyperparameters = request.hyperparameters
    if not model_id:
        api_logger.error("fit_save endpoint need model_id")
        raise HTTPException(status_code=400, detail="Требуется ID модели")
    elif model_id in DB["models"]:
        api_logger.error("not unique model_id")
        raise HTTPException(
            status_code=400, detail="Модель с таким ID уже загружена")
    if qdrant_client.collection_exists(collection_name=f"{model_id}"):
        api_logger.error("not unique qdrant collection")
        raise HTTPException(
            status_code=400, detail="Коллекция qdrant для данной модели уже существует")
    model_type = request.ml_model_type
    if model_type == "tf-idf":
        if 'ngram_range' in hyperparameters:
            hyperparameters['ngram_range'] = tuple(
                hyperparameters['ngram_range'])
        model = TfidfVectorizer(**request.hyperparameters)
    else:
        api_logger.error("unexpected model type")
        raise HTTPException(
            status_code=400, detail="Поддерживаемые типы моделей: tf-idf")
    cur_ds_nm = request.dataset_nm
    if cur_ds_nm not in DB["datasets"]:
        api_logger.error("unexpected dataset")
        raise HTTPException(
            status_code=404, detail="Датасет с таким именем не был загружен")
    cur_ds = DB["datasets"][cur_ds_nm]
    try:
        vectorized_contexts = model.fit_transform(cur_ds['context'])
        api_logger.info("model trained")
    except ValueError as exc:
        api_logger.error("incorrect train data")
        raise HTTPException(
            status_code=400, detail="Некорректные данные для обучения") from exc
    save_vectors_batch(
        qdrant_client, cur_ds['context'], vectorized_contexts, model_id)
    DB["models"][model_id] = {"type": model_type,
                              "model": model, "train_data": cur_ds['context'], "is_loaded": False,
                              "hyperparameters": hyperparameters, "dataset": cur_ds_nm}
    return MutlipleApiResponse(root=[{'message': f"Данные преобразованы и загружены в qdrant \
                                      с помощью модели '{model_id}'"}])


@router.post("/load_model", response_model=MutlipleApiResponse,
             description="Загрузка модели в RAM (TODO)")
async def load_model(request: Annotated[LoadRequest, 'Модель для загрузки датасета в память']):
    """Загрузка модели в память"""
    global DB
    model_id = request.model_id
    if model_id not in DB["models"]:
        api_logger.error("Model '%s' not in db", model_id)
        raise HTTPException(
            status_code=404, detail=f"Модель '{model_id}' не обучена")
    DB["models"][model_id]['is_loaded'] = True
    api_logger.info("Model '%s' loaded", model_id)
    return MutlipleApiResponse(root=[{'message': f"Модель '{model_id}' загружена"}])


@router.post("/unload_model", response_model=MutlipleApiResponse,
             description="Выгрузка моделей из RAM (TODO)")
async def unload_model():
    """Выгрузка модели"""
    global DB
    res = []
    cntr = 0
    for id_, vals in DB["models"].items():
        if vals['is_loaded'] is True:
            DB["models"][id_]['is_loaded'] = False
            res.append({'message': f"Model '{id_}' unloaded"})
            api_logger.info("Model '%s' unloaded", id_)
            cntr += 1
    if cntr == 0:
        api_logger.error("No model is currently loaded")
        raise HTTPException(
            status_code=400, detail="No model is currently loaded")
    return MutlipleApiResponse(root=res)


@router.post("/find_context", response_model=FindCntxtsResponse,
             description='Поиск контекста для вопроса')
async def find_context(request: Annotated[PredictRequest,
                                          "Вопрос для поиска подходящего контекста"]):
    """Поиск контекста"""
    model_id = request.model_id
    question = request.question
    global DB
    if model_id not in DB["models"] or not DB["models"][model_id]["is_loaded"]:
        api_logger.error("%s not loaded", model_id)
        raise HTTPException(
            status_code=404, detail=f"Модель {model_id} не загружена"
        )
    model = DB["models"][model_id]["model"]
    try:
        vectorized_question = model.transform([question])
    except ValueError as exc:
        api_logger.error("incorrect question data")
        raise HTTPException(
            status_code=400, detail="Некорректные данные для преобразования"
        ) from exc

    found_texts = search_similar_texts(
        qdrant_client, vectorized_question, model_id)
    rs = []
    for text in found_texts:
        source_text = text['source_text']
        score = text['score']
        point_id = text['point_id']
        rs.append({'model_id': model_id, 'context': source_text,
                  'score': score, 'point_id': point_id})
    api_logger.info("contexts found")
    return FindCntxtsResponse(root=rs)


@router.post("/quality_test", response_model=AccuracyResponse,
             description='Оценка точности и скорости работы модели')
async def quality_test(request: Annotated[CheckRequest,
                                          "Модель данных для тестирования датасета"]):
    """Тестирование модели"""
    global DB
    model_id = request.model_id
    model = DB["models"][model_id]["model"]
    cur_ds_nm = DB["models"][model_id]["dataset"]
    threshold = request.threshold
    cur_ds = DB["datasets"][cur_ds_nm][:threshold]
    try:
        res: dict[str, float | List[float]] = check_questions(
            qdrant_client, cur_ds, model, model_id)
    except Exception as exc:
        api_logger.error("quality_test unexpected error")
        raise HTTPException(
            status_code=500, detail="Ошибка при тестировании модели") from exc
    api_logger.info("quality checked")
    return res


@router.get("/get_datasets", response_model=DsListResponse,
            description='Список загруженных датасетов')
async def get_datasets():
    """Список загруженных датасетов"""
    global DB
    return {"datasets_nm": DB["datasets"].keys()}


# вывести гиперпараметры
@router.get("/list_models", response_model=ModelsListResponse,
            description='Список загруженных и обученных моделей')
async def list_models():
    """Список загруженных и обученных моделей"""
    global DB
    model_list = [
        {"model_id": model_id,
         "type": data["type"],
         "hyperparameters": data["hyperparameters"]}
        for model_id, data in DB["models"].items()
    ]
    return ModelsListResponse(root=[{'models': model_list}])


@router.delete("/remove/{model_id}", response_model=MutlipleApiResponse,
               description="Удаление модели по ее id")
async def remove(model_id: str):
    """Удаление модели по id"""
    global DB
    global qdrant_client
    if model_id not in DB["models"]:
        api_logger.info("model %s not loaded but tried to delete", model_id)
        raise HTTPException(status_code=404, detail="Model not found")
    del DB["models"][model_id]
    if qdrant_client.collection_exists(collection_name=f"{model_id}"):
        qdrant_client.delete_collection(collection_name=f"{model_id}")
    return MutlipleApiResponse(root=[{'message': f"Model '{model_id}' removed"}])


@router.delete("/remove_all", response_model=MutlipleApiResponse,
               description="Удаление всех моделей и коллекций в qdrant")
def remove_all():
    """Удаление всех моделей и коллекций в qdrant"""
    global DB
    messages = [{'message': f"Model '{model_id}' removed"}
                for model_id in DB["models"]
                ]
    for model_id in DB["models"]:
        if qdrant_client.collection_exists(collection_name=f"{model_id}"):
            qdrant_client.delete_collection(collection_name=f"{model_id}")
    DB["models"].clear()
    return MutlipleApiResponse(root=messages)
