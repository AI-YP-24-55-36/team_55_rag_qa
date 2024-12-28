import pytest
from fastapi.testclient import TestClient
from main import app
import pandas as pd
from unittest.mock import Mock

client = TestClient(app)

@pytest.fixture
def mock_db():
    return {
        "datasets": {},
        "models": {}
    }

@pytest.fixture
def mock_qdrant_client():
    return Mock()

def test_load_dataset():
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

    # Тест на повторную загрузку
    response = client.post("/api/v1/models/load_dataset", json=test_data)
    assert response.status_code == 400
    assert "Датасет с таким названием уже был загружен" in response.json()["detail"]

def test_fit_save():
    # Сначала загрузим датасет
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
    assert "Данные преобразованы и загружены в qdrant с помощью модели 'model1'" in response.json()[0]["message"]

    # Тест на повторное обучение с тем же ID
    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 400
    assert "Модель с таким ID уже загружена" in response.json()["detail"]

def test_load_unload_model():
    # Загрузка модели
    response = client.post("/api/v1/models/load_model", json={"model_id": "model1"})
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
    response = client.get("/api/v1/models/get_datasets")
    assert response.status_code == 200
    assert "datasets_nm" in response.json()

def test_list_models():
    response = client.get("/api/v1/models/list_models")
    assert response.status_code == 200

def test_remove_model():
    # Удаление существующей модели
    response = client.delete("/api/v1/models/remove/model1")
    assert response.status_code == 200
    assert "Model 'model1' removed" in response.json()[0]["message"]

    # Удаление несуществующей модели
    response = client.delete("/api/v1/models/remove/nonexistent_model")
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]

def test_remove_all_models():
    # Удаление всех моделей
    response = client.delete("/api/v1/models/remove_all")
    assert response.status_code == 200
    
    # Попытка удалить, когда моделей нет
    
    # response = client.delete("/api/v1/models/remove_all")
    # assert response.status_code == 404
    # assert "Models not found" in response.json()["detail"]


# Тестирование на большом датасете
def test_load_full_dataset():
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

    # Тест на повторную загрузку
    response = client.post("/api/v1/models/load_dataset", json=test_data)
    assert response.status_code == 400
    assert "Датасет с таким названием уже был загружен" in response.json()["detail"]

def test_full_fit_save():
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
    assert "Данные преобразованы и загружены в qdrant с помощью модели 'model1'" in response.json()[0]["message"]

    # Тест на повторное обучение с тем же ID
    response = client.post("/api/v1/models/fit_save", json=fit_request)
    assert response.status_code == 400
    assert "Модель с таким ID уже загружена" in response.json()["detail"]

def test_full_load_unload_model():
    # Загрузка модели
    response = client.post("/api/v1/models/load_model", json={"model_id": "model1"})
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
    response = client.get("/api/v1/models/get_datasets")
    assert response.status_code == 200
    assert "datasets_nm" in response.json()

def test_full_list_models():
    response = client.get("/api/v1/models/list_models")
    assert response.status_code == 200



def test_check():
    request = {
        "model_id": "model1"
    }
    response = client.post("/api/v1/models/quality_test", json=request)
    assert response.status_code == 200

def test_full_remove_model():
    # Удаление существующей модели
    response = client.delete("/api/v1/models/remove/model1")
    assert response.status_code == 200
    assert "Model 'model1' removed" in response.json()[0]["message"]

    # Удаление несуществующей модели
    response = client.delete("/api/v1/models/remove/nonexistent_model")
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]

def test_full_remove_all_models():
    # Удаление всех моделей
    response = client.delete("/api/v1/models/remove_all")
    assert response.status_code == 200
    