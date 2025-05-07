import datetime
import logging
import sys
import time
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from beir.retrieval.models import SentenceBERT
from sentence_transformers import CrossEncoder
from qdrant_client import models
from qdrant_client.models import (
    Distance,
    Modifier,
    MultiVectorConfig,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams
)
from tqdm import tqdm

from log_output import Tee
from load_config import load_config

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
OUTPUT_DIR = BASE_DIR / config["paths"]["output_dir"]
EMBEDDINGS_DIR = BASE_DIR / config["paths"]["embeddings_dir"]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Tee(f"{OUTPUT_DIR}/log_{timestamp}.txt")

logger = logging.getLogger('hybrid')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(f'{LOGS_DIR}/hybrid.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)

# Загружаем модель для реранкинга
reranker_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# функция удаляет коллекцию с таким именем, если она существует
def clear_existing_collections(client):
    collections = client.get_collections().collections
    for collection in collections:
        client.delete_collection(collection.name)
        print(f"Collection {collection.name} has been cleared")

# функция для создания коллекции
def create_hybrid_collection(client, collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=768,
                distance=Distance.COSINE
            ),
            "colbertv2.0": VectorParams(
                size=128,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator="max_sim")
            ),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF
            )
        }
    )
    logger.info(f"Создана коллекция {collection_name}, готова к заполнению")
    print(f"Создана коллекция {collection_name}, готова к заполнению")

dense_embeddings = np.memmap('embeddings/dense_embeddings.memmap', dtype='float32', mode='r').reshape(-1, 768)
colbert_embeddings = np.memmap('embeddings/colbert_embeddings.memmap', dtype='float32', mode='r').reshape(-1, 256, 128)
with open('embeddings/sparse_embeddings.pkl', 'rb') as f:
    sparse_embeddings = pickle.load(f)

def build_point_from_files(
    item,
    idx,
    sparse_embeddings,
    dense_embeddings,
    colbert_embeddings
):
    # Получаем эмбеддинги по индексу
    sparse_embedding = sparse_embeddings[idx]
    dense_embedding = dense_embeddings[idx].tolist()
    colbert_embedding = colbert_embeddings[idx].tolist()

    return models.PointStruct(
        id=item["id"],
        payload=item,
        vector={
            "bm25": {
                "values": sparse_embedding.values.tolist() if sparse_embedding else [],
                "indices": sparse_embedding.indices.tolist() if sparse_embedding else []
            },
            "dense": dense_embedding,
            "colbertv2.0": colbert_embedding
        }
    )

# загрузка поинтов батчами
def upload_points_in_batches(client, collection_name, points, batch_size=50):
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upload_points(
            collection_name=collection_name,
            points=batch,
        )
        print(f"Загружено {i + len(batch)} из {len(points)} документов")

# создание и загрузка коллекции
def upload_hybrid_data(client, collection_name: str, data):
    """Загрузка данных в Qdrant с поддержкой гибридного поиска (BM25 + Dense + ColBERT)"""
    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с гибридным поиском")
    clear_existing_collections(client)
    create_hybrid_collection(client, collection_name)
    # bm25_model, dense_model, colbert_model = load_embedding_models()
    logger.info(f"⏳ Создание точек загрузки {collection_name}")
    print(f"⏳ Создание точек загрузки  {collection_name}")
    points = []
    for idx, item in tqdm(enumerate(data)):
        point = build_point_from_files(item, idx, sparse_embeddings, dense_embeddings, colbert_embeddings)
        points.append(point)
    print(f"Создано {len(points)} points")
    upload_points_in_batches(client, collection_name, points)
    logger.info(f"✅ Данные успешно загружены в коллекцию {collection_name}")
    print(f"✅ Данные успешно загружены в коллекцию {collection_name}")


#  Для теста модели
def load_embedding():
    return {
        "bm25": SparseTextEmbedding("Qdrant/bm25"),
        "dense": SentenceBERT("msmarco-distilbert-base-tas-b"),
        "colbert": LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    }

def encode_query(query_text, models):
    sparse_vector = list(models["bm25"].query_embed(query_text))
    sparse_embedding = sparse_vector[0] if sparse_vector else None
    dense_embedding = models["dense"].encode_corpus([{"text": query_text}], convert_to_tensor=False)[0]
    colbert_embedding = list(models["colbert"].embed(query_text))[0]

    return sparse_embedding, dense_embedding, colbert_embedding


def run_hybrid_search(client, collection_name, sparse_embedding, dense_embedding, colbert_embedding, top_k):
    prefetch = [
        models.Prefetch(query=dense_embedding, using="dense", limit=20),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_embedding.indices.tolist() if sparse_embedding else [],
                values=sparse_embedding.values.tolist() if sparse_embedding else []
            ),
            using="bm25",
            limit=20
        ),
    ]

    start_time = time.time()
    search_results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=colbert_embedding,
        using="colbertv2.0",
        with_payload=True,
        limit=top_k
    )
    end_time = time.time()

    query_time = end_time - start_time
    found_contexts = [(point.payload.get('context', ''), point.score) for point in search_results.points]

    return found_contexts, query_time

def evaluate_accuracy(found_contexts, true_context, top_k_values, results, stage):
    if not found_contexts:
        return

    # Проверим, является ли каждый элемент кортежем
    if not all(isinstance(item, (tuple, list)) and len(item) == 2 for item in found_contexts):
        raise ValueError(f"Неверный формат found_contexts: ожидался список кортежей (context, score), получено: {found_contexts}")

    for k in top_k_values:
        results["accuracy"][stage][k]["total"] += 1
        top_k_contexts = [ctx for ctx, _ in found_contexts[:k]]
        if true_context in top_k_contexts:
            results["accuracy"][stage][k]["correct"] += 1


def update_speed_metrics(results):
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)
    del results["speed"]["query_times"]


def log_final_metrics(results, top_k_values):
    for stage in ["before_rerank", "after_rerank"]:
        for k in top_k_values:
            correct = results["accuracy"][stage][k]["correct"]
            total = results["accuracy"][stage][k]["total"]
            accuracy = correct / total if total > 0 else 0
            results["accuracy"][stage][k]["accuracy"] = accuracy
            logger.info(f"Hybrid Search {stage.replace('_', ' ')} (top-{k}): {accuracy:.4f} ({correct}/{total})")

    logger.info(f"Hybrid Search Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")


def benchmark_hybrid_rerank(client, collection_name, test_data, top_k_values=[1, 3], reranker=None):
    print(f"\n🔍 Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")

    results = {
        "speed": {
            "avg_time": 0, "median_time": 0, "max_time": 0, "min_time": 0, "query_times": []
        },
        "accuracy": {
            "before_rerank": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values},
            "after_rerank": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
        }
    }

    max_top_k = max(top_k_values)
    total_queries = len(test_data)

    logger.info(f"Оценка производительности Hybrid Search для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности гибридного поиска...")

    embedding_models = load_embedding()
    progress_bar = tqdm(total=total_queries, desc="Обработка запросов", unit="запрос")

    for _, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        sparse_embedding, dense_embedding, colbert_embedding = encode_query(query_text, embedding_models)
        found_contexts, query_time = run_hybrid_search(client, collection_name, sparse_embedding, dense_embedding, colbert_embedding, max_top_k)
        results["speed"]["query_times"].append(query_time)

        evaluate_accuracy(found_contexts, true_context, top_k_values, results, stage="before_rerank")

        if reranker:
            reranked_contexts = reranker(query_text, found_contexts)
            evaluate_accuracy(reranked_contexts, true_context, top_k_values, results, stage="after_rerank")

        progress_bar.update(1)

    progress_bar.close()

    update_speed_metrics(results)
    log_final_metrics(results, top_k_values)

    print(f"✅ Оценка производительности Hybrid Search завершена для коллекции '{collection_name}'")
    return results

def reranker(query, candidates):
    texts = [(query, context) for context, _ in candidates]
    scores = reranker_model.predict(texts)

    # Добавляем новые оценки и сортируем по ним
    reranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Возвращаем кортежи: (context, new_score)
    return [(context, score) for (context, _), score in reranked_results]


def print_comparison(results_without_rerank, results_with_rerank, top_k_values=[1, 3]):
    print("\n📊 Сравнение результатов Hybrid Search с реранкингом и без него:\n")

    print("⏳ Время выполнения запроса:")
    print(f"  - Без реранкинга: {results_without_rerank['speed']['avg_time'] * 1000:.2f} мс")
    print(f"  - С реранкингом: {results_with_rerank['speed']['avg_time'] * 1000:.2f} мс")

    print("\n🎯 Точность поиска (Accuracy):")
    for k in top_k_values:
        acc_before = results_without_rerank["accuracy"]["before_rerank"][k]["accuracy"]
        acc_after = results_with_rerank["accuracy"]["after_rerank"][k]["accuracy"]

        print(f"  - Top-{k}:")
        print(f"    - Без реранкинга: {acc_before:.4f}")
        print(f"    - С реранкингом: {acc_after:.4f}")


def visualize_results_rerank(results_without_rerank, results_with_rerank, top_k_values=[1, 3],
                             title_prefix="Сравнение для гибридного поиска с реранкингом и без", save_dir=f"{GRAPHS_DIR}"):

    print(f"\n📊 Создание визуализаций результатов реранкинга...")
    logger.info("Создание визуализаций результатов реранкинга")

    print(save_dir)

    # Создаем директорию для сохранения графиков, если она не существует
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Генерация метки времени
    timestr = time.strftime("%Y%m%d_%H%M%S")  # Убедимся, что символы в имени файла допустимы

    # --- 1️⃣ Визуализация времени выполнения ---
    plt.figure(figsize=(10, 5))

    speeds = [
        results_without_rerank['speed']['avg_time'] * 1000,  # в миллисекундах
        results_with_rerank['speed']['avg_time'] * 1000
    ]

    bar_width = 0.8 /2
    n_groups = len(top_k_values)
    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, 2))
    labels = ["Без реранкинга", "С реранкингом"]  # Подписи столбцов

    # # Создаём график времени

    # Построение столбчатой диаграммы
    plt.bar(
        index,
        speeds,
        bar_width,
        color=colors,  # Цвета столбцов
        edgecolor='black',  # Цвет границы столбцов
        linewidth=0.5,  # Толщина границы столбцов
    )

    # Добавляем значения над столбцами
    for i, v in enumerate(speeds):
        if v > 0:
            plt.text(
                index[i],
                v + 1,
                f"{v:.1f}",
                ha='center',
                va='bottom',
                fontsize=6,
            )
    # Настройка осей
    plt.xticks(index, labels)
    plt.ylabel("Время (мс)")
    plt.title(f"{title_prefix}: Время поиска")

    # Сетка для оси Y
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Сохранение графика (до plt.show())
    speed_save_path = f"{save_dir}/speed_comparison_{timestr}_hybrid.png"
    plt.savefig(speed_save_path, dpi=300, bbox_inches='tight')

    # --- 2️⃣ Визуализация точности поиска ---
    plt.figure(figsize=(10, 5))
    acc_before = [results_without_rerank["accuracy"]["before_rerank"][k]["accuracy"] for k in top_k_values]
    acc_after = [results_with_rerank["accuracy"]["after_rerank"][k]["accuracy"] for k in top_k_values]

    # Настройки для графика
    bar_width = 0.8 / 2
    n_groups = len(top_k_values)
    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, 2))
    labels = ["Без реранкинга", "С реранкингом"]

    # Построение столбчатой диаграммы для точности
    plt.bar(
        index - bar_width / 2,
        acc_before,
        bar_width,
        label=labels[0],
        color=colors[0],
        edgecolor='black',
        linewidth=0.5,
    )

    plt.bar(
        index + bar_width / 2,
        acc_after,
        bar_width,
        label=labels[1],
        color=colors[1],
        edgecolor='black',
        linewidth=0.5,
    )

    # Добавляем значения над столбцами
    for i, (v_before, v_after) in enumerate(zip(acc_before, acc_after)):
        if v_before > 0:
            plt.text(
                index[i] - bar_width / 2,
                v_before + 0.01,
                f"{v_before:.2f}",
                ha='center',
                va='bottom',
                fontsize=6,
            )
        if v_after > 0:
            plt.text(
                index[i] + bar_width / 2,
                v_after + 0.01,
                f"{v_after:.2f}",
                ha='center',
                va='bottom',
                fontsize=6,
            )

    # Настройка осей
    plt.xticks(index, [f"Top-{k}" for k in top_k_values])
    plt.ylabel("Точность (Accuracy)")
    plt.title(f"{title_prefix}: Точность поиска")

    # Легенда и сетка
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    accuracy_save_path = f"{save_dir}/accuracy_comparison_{timestr}_hybrid.png"
    plt.savefig(accuracy_save_path, dpi=300, bbox_inches='tight')