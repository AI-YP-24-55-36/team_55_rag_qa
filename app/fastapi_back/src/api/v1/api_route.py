from fastapi import APIRouter, HTTPException
from http import HTTPStatus
from qdrant_client import QdrantClient, models
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant.load_qdrant import save_vectors_batch, search_similar_texts, check_questions
from .schemas import *

DB = {"datasets": {}, "models": {}}

router = APIRouter(prefix="/api/v1/models")
qdrant_client = QdrantClient(url="http://localhost:6333", timeout=1000)


# API endpoints
@router.post("/load_dataset", response_model=MutlipleApiResponse, status_code=HTTPStatus.CREATED,
             description='Загрузка датасета')
async def fit(request: DatasetRequest):
    global DB
    rs = []
    # TODO Работа с несколькими датасетами
    if DB['datasets']:
        del DB['datasets']
        del DB['models']
    DB['datasets'] = {}
    DB['models'] = {}
    datasets = request.datasets
    for dataset_nm, dataset in datasets.items():
        if dataset_nm in DB['datasets']:
            raise HTTPException(
                status_code=400, detail="Датасет с таким названием уже был загружен")
        DB['datasets'][dataset_nm] = pd.DataFrame(dataset)
        rs.append({'message': f"Датасет '{dataset_nm}' загружен"})
    return MutlipleApiResponse(root=rs)

'''Обучение на датасете и загрузка в qdrant'''


@router.post("/fit_save", response_model=MutlipleApiResponse, status_code=HTTPStatus.CREATED,
             description='Обучение на датасете и загрузка в qdrant')
async def fit(request: FitRequestList):
    global DB
    global qdrant_client
    # TODO Обучение нескольких моделей в 1 запросе
    request = request.root[0]
    model_id = request.model_id
    hyperparameters = request.hyperparameters
    if not model_id:
        raise HTTPException(status_code=400, detail="Требуется ID модели")
    elif model_id in DB["models"]:
        raise HTTPException(
            status_code=400, detail="Модель с таким ID уже загружена")
    if qdrant_client.collection_exists(collection_name=f"{model_id}"):
        raise HTTPException(
            status_code=400, detail="Коллекция qdrant для данной модели уже существует")
    model_type = request.ml_model_type
    if model_type == "tf-idf":
        if 'ngram_range' in hyperparameters:
            hyperparameters['ngram_range'] = tuple(
                hyperparameters['ngram_range'])
        model = TfidfVectorizer(**request.hyperparameters)
    else:
        raise HTTPException(
            status_code=400, detail="Поддерживаемые типы моделей: tf-idf")
    cur_ds_nm = request.dataset_nm
    if cur_ds_nm not in DB["datasets"]:
        raise HTTPException(
            status_code=404, detail="Датасет с таким именем не был загружен")
    cur_ds = DB["datasets"][cur_ds_nm]
    try:
        vectorized_contexts = model.fit_transform(cur_ds['context'])
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Некорректные данные для обучения")
    save_vectors_batch(
        qdrant_client, cur_ds['context'], vectorized_contexts, model_id)
    DB["models"][model_id] = {"type": model_type,
                              "model": model, "train_data": cur_ds['context'], "is_loaded": False,
                              "hyperparameters": hyperparameters, "dataset": cur_ds_nm}
    return MutlipleApiResponse(root=[{'message': f"Данные преобразованы и загружены в qdrant с помощью модели '{model_id}'"}])


@router.post("/load_model", response_model=MutlipleApiResponse,
             description="Загрузка модели в RAM (TODO)")
async def load_model(request: LoadRequest):
    global DB
    model_id = request.model_id
    if model_id not in DB["models"]:
        raise HTTPException(
            status_code=404, detail=f"Модель '{model_id}' не обучена")
    DB["models"][model_id]['is_loaded'] = True
    return MutlipleApiResponse(root=[{'message': f"Модель '{model_id}' загружена"}])


@router.post("/unload_model", response_model=MutlipleApiResponse,
             description="Выгрузка моделей из RAM (TODO)")
async def unload_model(
):
    global DB
    res = []
    cntr = 0
    for id, vals in DB["models"].items():
        if vals['is_loaded'] is True:
            DB["models"][id]['is_loaded'] = False
            res.append({'message': f"Model '{id}' unloaded"})
            cntr += 1
    if cntr == 0:
        raise HTTPException(
            status_code=400, detail="No model is currently loaded")
    return MutlipleApiResponse(root=res)


@router.post("/find_context", response_model=FindCntxtsResponse,
             description='Поиск контекста для вопроса')
async def find_context(request: PredictRequest):
    model_id = request.model_id
    question = request.question
    global DB
    if model_id not in DB["models"] or not DB["models"][model_id]["is_loaded"]:
        raise HTTPException(
            status_code=404, detail=f"Модель {model_id} не загружена"
        )
    model = DB["models"][model_id]["model"]
    try:
        vectorized_question = model.transform([question])
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Некорректные данные для преобразования"
        )

    found_texts = search_similar_texts(
        qdrant_client, vectorized_question, model_id)
    rs = []
    for text in found_texts:
        source_text = text['source_text']
        score = text['score']
        point_id = text['point_id']
        rs.append({'model_id': model_id, 'context': source_text,
                  'score': score, 'point_id': point_id})
    return FindCntxtsResponse(root=rs)


@router.post("/quality_test", response_model=AccuracyResponse,
             description='Оценка точности и скорости работы модели')
async def quality_test(request: CheckRequest):
    global DB
    model_id = request.model_id
    model = DB["models"][model_id]["model"]
    cur_ds_nm = DB["models"][model_id]["dataset"]
    threshold = request.threshold
    cur_ds = DB["datasets"][cur_ds_nm][:threshold]
    res: dict[str, float | List[float]] = check_questions(
        qdrant_client, cur_ds, model, model_id)
    return res


@router.get("/get_datasets", response_model=DsListResponse,
            description='Получить список загруженных датасетов')
async def get_datasets():
    global DB
    return {"datasets_nm": DB["datasets"].keys()}

# TODO вывести гиперпараметры
@router.get("/list_models", response_model=ModelsListResponse,
            description='Получить список загруженных и обученных моделей')
async def list_models():
    global DB
    model_list = [
        {"model_id": model_id,
         "type": data["type"],
         "hyperparameters": data["hyperparameters"]}
        for model_id, data in DB["models"].items()
    ]
    return ModelsListResponse(root=[{'models': model_list}])


@router.delete("/remove/{model_id}", response_model=MutlipleApiResponse)
async def remove(model_id: str):
    global DB
    global qdrant_client
    if model_id not in DB["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    del DB["models"][model_id]
    if qdrant_client.collection_exists(collection_name=f"{model_id}"):
        qdrant_client.delete_collection(collection_name=f"{model_id}")
    return MutlipleApiResponse(root=[{'message': f"Model '{model_id}' removed"}])


@router.delete("/remove_all", response_model=MutlipleApiResponse)
def remove_all():
    global DB
    messages = [{'message': f"Model '{model_id}' removed"}
                for model_id in DB["models"]
                ]
    for model_id in DB["models"].keys():
        if qdrant_client.collection_exists(collection_name=f"{model_id}"):
            qdrant_client.delete_collection(collection_name=f"{model_id}")
    DB["models"].clear()
    return MutlipleApiResponse(root=messages)
