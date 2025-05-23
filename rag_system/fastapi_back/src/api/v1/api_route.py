from http import HTTPStatus
from typing import List
import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

from src.qdrant.load_qdrant import save_vectors_batch, search_similar_texts, check_questions
from src.logger import api_logger
from src.config import API_KEY_GROQ

from src.api.v1.schemas import (
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
    FindAnswerResponse
)


db = {"datasets": {}, "models": {}}
router = APIRouter(prefix="/api/v1/models")
qd_client = QdrantClient(url="http://qdrant:6333", timeout=1000)


# API endpoints
@router.post("/load_dataset", response_model=MutlipleApiResponse, status_code=HTTPStatus.CREATED,
             description='Загрузка датасета')
async def fit(request: Annotated[DatasetRequest, 'Датасеты в формате массива списков']):
    """Функция обучения модели"""
    DB = db
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
    DB = db
    qdrant_client = qd_client
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
    DB = db
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
    DB = db
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
    DB = db
    qdrant_client = qd_client
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


@router.post("/find_answer",
             description='Поиск ответа на вопрос')
async def find_answer(request: Annotated[PredictRequest, "Вопрос для поиска ответа"]):
    """Поиск ответа"""
    try:
        context_response = await find_context(request)
        top_contexts = context_response.root[:3]
        if not top_contexts:
            raise HTTPException(
                status_code=404,
                detail="Не найдено подходящих контекстов"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Ошибка при поиске контекстов: {str(e)}'
            ) from e

    contexts_text = "\n\n".join(
        [f"Контекст {i+1} (score={ctx.score:.2f}): {ctx.context}"
         for i, ctx in enumerate(top_contexts)]
    )
    # Ты - помощник, который отвечает на вопрос на основе предоставленных контекстов и самого вопроса.
    # Ты берешь информацию только из контекстов.
    # Если ответа нет в контекстах, скажи об этом.
    system_prompt = """You are a helper who answers a question based on the contexts provided and the question itself.
    You only take information from the contexts.
    If the answer is not in the contexts, say so."""

    user_prompt = f"""Question: {request.question}
    Contexts:
    {contexts_text}
    Answer as accurately as possible using only these contexts."""
    # Ответь максимально точно, используя только эти контексты

    try:
        client = Groq(api_key=API_KEY_GROQ)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1024
        )

        llm_answer = chat_completion.choices[0].message.content

        # оценка ответа через другую модель
        # Оцени ответ по шкале от 1 до 5 по критериям:
        # 1. **Релевантность** (соответствие вопросу).
        # 2. **Точность** (использование контекста без ошибок).
        # 3. **Грамотность** (ясность и связность текста).
        # И напиши краткое описание почему получились такие оценки.
        # Выведи оценки в формате JSON
        evaluation_prompt = f"""
        Question: {request.question}
        Contexts: {contexts_text}
        Answer: {llm_answer}
        Approach the task as strictly as possible.
        Rate the answer on a scale from 1 to 5 based on the criteria:
        1. **Relevance** (correspondence to the question).
        2. **Accuracy** (use of context without errors).
        3. **Literacy** (clarity and coherence of the text).
        Provide a description of why such ratings were obtained, the description must be in Russian.
        Output the ratings in JSON format: {{"relevance": X, "accuracy": Y, "fluency": Z, "description": desc}}
        """

        system_eval_prompt = """You are an expert who checks the generative model for correct answers.
        Approach the task as strictly as possible."""
        evaluation_response = client.chat.completions.create(
            messages=[{"role": "system", "content": system_eval_prompt},
                      {"role": "user", "content": evaluation_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        metrics = json.loads(evaluation_response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при запросе к LLM: {str(e)}"
        ) from e

    return FindAnswerResponse(
        question=request.question,
        answer=llm_answer,
        used_contexts=[{"context": ctx.context, "score": ctx.score, "point_id": ctx.point_id, "model_id": ctx.model_id}
                       for ctx in top_contexts],
        metrics=metrics,
        model_used="llama3-70b-8192",
        model_judge="llama-3.3-70b-versatile"
    )


@router.post("/quality_test", response_model=AccuracyResponse,
             description='Оценка точности и скорости работы модели')
async def quality_test(request: Annotated[CheckRequest,
                                          "Модель данных для тестирования датасета"]):
    """Тестирование модели"""
    DB = db
    qdrant_client = qd_client
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
    DB = db
    return {"datasets_nm": DB["datasets"].keys()}


# вывести гиперпараметры
@router.get("/list_models", response_model=ModelsListResponse,
            description='Список загруженных и обученных моделей')
async def list_models():
    """Список загруженных и обученных моделей"""
    DB = db
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
    DB = db
    qdrant_client = qd_client
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
    DB = db
    qdrant_client = qd_client
    messages = [{'message': f"Model '{model_id}' removed"}
                for model_id in DB["models"]
                ]
    for model_id in DB["models"]:
        if qdrant_client.collection_exists(collection_name=f"{model_id}"):
            qdrant_client.delete_collection(collection_name=f"{model_id}")
    DB["models"].clear()
    return MutlipleApiResponse(root=messages)


if __name__ == "__main__":
    pass
