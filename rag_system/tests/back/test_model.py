from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fastapi_back.src.main import app

client = TestClient(app)


@pytest.fixture
def mock_db():
    """Функция создания дб"""
    return {
        "datasets": {},
        "models": {}
    }


@pytest.fixture
def mock_qdrant_client():
    """Функция подключения qdrant"""
    return Mock()


def test_load_dataset():
    """Функция создания тестовых данных"""
    test_data = {
        "datasets": {
            "test_dataset": [
                {"context": "text1", "question": "q1", "answer": "a1"},
                {"context": "text2", "question": "q2", "answer": "a2"}
            ]
        }
    }

    response = client.post("/api/v1/models/load_dataset", json=test_data)
    assert response.status_code == 201
    assert "Датасет 'test_dataset' загружен" in response.json()[0]["message"]


def test_fit_save():
    """Функция теста обученния модели"""
    dataset = {
        "datasets": {
            "test_dataset": [
                {"context": "text1", "question": "q1", "answer": "a1"},
                {"context": "text2", "question": "q2", "answer": "a2"}
            ]
        }
    }
    client.post("/api/v1/models/load_dataset", json=dataset)

    # Тест обучения модели
    fit_request = [{
        "model_id": "model1",
        "ml_model_type": "tf-idf",
        "dataset_nm": "test_dataset",
        "hyperparameters": {}
    }]

    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 201
    assert "Данные преобразованы и загружены в qdrant с помощью модели 'model1'" in response.json()[
        0]["message"]

    # Тест на повторное обучение с тем же ID
    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 400
    assert "Модель с таким ID уже загружена" in response.json()["detail"]


def test_load_unload_model():
    """Функция теста загрузки, выгрузки модели"""
    # Загрузка модели
    response = client.post("/api/v1/models/load_model",
                           json={"model_id": "model1"})
    assert response.status_code == 200
    assert "Модель 'model1' загружена" in response.json()[0]["message"]

    # Выгрузка модели
    response = client.post("/api/v1/models/unload_model")
    assert response.status_code == 200
    assert "Model 'model1' unloaded" in response.json()[0]["message"]

    # Попытка выгрузить, когда нет загруженных моделей
    response = client.post("/api/v1/models/unload_model")
    assert response.status_code == 400
    assert "No model is currently loaded" in response.json()["detail"]


def test_find_context():
    """Функция теста поиска контекста"""
    # Тест поиска контекста
    request = {
        "model_id": "model1",
        "question": "What is the meaning of life?"
    }

    # Тест с незагруженной моделью
    response = client.post("/api/v1/models/find_context", json=request)
    assert response.status_code == 404
    assert "Модель model1 не загружена" in response.json()["detail"]

    # Загрузим модель и попробуем снова
    client.post("/api/v1/models/load_model", json={"model_id": "model1"})
    response = client.post("/api/v1/models/find_context", json=request)
    assert response.status_code == 200


def test_get_datasets():
    """Функция теста получения датасета"""
    response = client.get("/api/v1/models/get_datasets")
    assert response.status_code == 200
    assert "datasets_nm" in response.json()


def test_list_models():
    """Функция теста получения списка моделей"""
    response = client.get("/api/v1/models/list_models")
    assert response.status_code == 200


def test_remove_model():
    """Функция теста удаления модели"""
    # Удаление существующей модели
    response = client.delete("/api/v1/models/remove/model1")
    assert response.status_code == 200
    assert "Model 'model1' removed" in response.json()[0]["message"]

    # Удаление несуществующей модели
    response = client.delete("/api/v1/models/remove/nonexistent_model")
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]


def test_remove_all_models():
    """Функция теста удаления всех моделей"""
    # Удаление всех моделей
    response = client.delete("/api/v1/models/remove_all")
    assert response.status_code == 200


# Тестирование на большом датасете
def test_load_full_dataset():
    """Функция теста большого датасета"""
    df = pd.read_csv('full_dataset.csv')
    df = df.fillna('')
    test_data = {
        "datasets": {
            "full_dataset": df.to_dict('records')
        }
    }

    response = client.post("/api/v1/models/load_dataset", json=test_data)
    assert response.status_code == 201
    assert "Датасет 'full_dataset' загружен" in response.json()[0]["message"]


def test_full_fit_save():
    """Функция теста полного цикла обучения"""
    # Сначала загрузим датасет
    df = pd.read_csv('full_dataset.csv')
    df = df.fillna('')
    dataset = {
        "datasets": {
            "full_dataset": df.to_dict('records')
        }
    }
    client.post("/api/v1/models/load_dataset", json=dataset)
    # Тест обучения модели
    fit_request = [{
        "model_id": "model1",
        "ml_model_type": "tf-idf",
        "dataset_nm": "full_dataset",
        "hyperparameters": {
            'stop_words': 'english',
            'ngram_range': [1, 2],
            'max_df': 0.85,
            'sublinear_tf': True}
    }]

    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 201
    assert "Данные преобразованы и загружены в qdrant с помощью модели 'model1'" in response.json()[
        0]["message"]

    # Тест на повторное обучение с тем же ID
    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 400
    assert "Модель с таким ID уже загружена" in response.json()["detail"]


def test_full_load_unload_model():
    """Функция теста полного цикла загрузки выгрузки модели"""
    # Загрузка модели
    response = client.post("/api/v1/models/load_model",
                           json={"model_id": "model1"})
    assert response.status_code == 200
    assert "Модель 'model1' загружена" in response.json()[0]["message"]

    # Выгрузка модели
    response = client.post("/api/v1/models/unload_model")
    assert response.status_code == 200
    assert "Model 'model1' unloaded" in response.json()[0]["message"]

    # Попытка выгрузить, когда нет загруженных моделей
    response = client.post("/api/v1/models/unload_model")
    assert response.status_code == 400
    assert "No model is currently loaded" in response.json()["detail"]


def test_full_find_context():
    """Функция теста получения контекста на большом объеме"""
    # Тест поиска контекста
    request = {
        "model_id": "model1",
        "question": "What is the meaning of life?"
    }

    # Тест с незагруженной моделью
    response = client.post("/api/v1/models/find_context", json=request)
    assert response.status_code == 404
    assert "Модель model1 не загружена" in response.json()["detail"]

    # Загрузим модель и попробуем снова
    client.post("/api/v1/models/load_model", json={"model_id": "model1"})
    response = client.post("/api/v1/models/find_context", json=request)
    assert response.status_code == 200


def test_full_get_datasets():
    """Функция теста получения полного датасета"""
    response = client.get("/api/v1/models/get_datasets")
    assert response.status_code == 200
    assert "datasets_nm" in response.json()


def test_full_list_models():
    """Функция теста получения полного списка моделей"""
    response = client.get("/api/v1/models/list_models")
    assert response.status_code == 200


def test_check():
    """Функция теста состояния модели"""
    request = {
        "model_id": "model1",
        "threshold": 1000
    }
    response = client.post("/api/v1/models/quality_test", json=request)
    assert response.status_code == 200


def test_full_remove_model():
    """Функция теста удаления модели по id"""
    # Удаление существующей модели
    response = client.delete("/api/v1/models/remove/model1")
    assert response.status_code == 200
    assert "Model 'model1' removed" in response.json()[0]["message"]

    # Удаление несуществующей модели
    response = client.delete("/api/v1/models/remove/nonexistent_model")
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]


def test_full_remove_all_models():
    """Функция теста удаления всех моделей"""
    # Удаление всех моделей
    response = client.delete("/api/v1/models/remove_all")
    assert response.status_code == 200
