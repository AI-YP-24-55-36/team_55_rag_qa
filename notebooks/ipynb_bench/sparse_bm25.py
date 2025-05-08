import datetime
import logging
import sys
import time
from tqdm import tqdm
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import models
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

# формат логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def create_coll(client, collection_name):
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        client.delete_collection(collection_name)
        logger.info(f"Коллекция {collection_name} удалена")

    # создание коллекцию с BM25-индексом
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=0, )  # отключение построение графа
    ),

    logger.info(f"Коллекция {collection_name} создана с поддержкой BM25")



def upload_bm25_data(client, collection_name, data):
    """Загрузка данных в Qdrant с использованием встроенного BM25"""

    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием BM25")
    # проверка, существует ли коллекция
    create_coll(client, collection_name)
    # инициализация модели
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    points = []
    # создание поинтов
    for item in data:
        vector = list(bm25_embedding_model.query_embed(item["context"]))
        if vector:
            sparse_embedding = vector[0]
            points.append(
                models.PointStruct(
                    id=item["id"],
                    payload=item,
                    vector={
                        "bm25": {
                            "values": sparse_embedding.values.tolist(),
                            "indices": sparse_embedding.indices.tolist()
                        }
                    }
                )
            )

    # загрузка поинтов
    client.upload_points(
        collection_name=collection_name,
        points=points
    )

    logger.info(f"Загрузка данных завершена для коллекции {collection_name}")


def init_results(top_k_values):
    return {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
    }


def evaluate_accuracy(accuracy_results, found_contexts, true_context, top_k_values, query_text, query_idx):
    for k in top_k_values:
        accuracy_results[k]["total"] += 1
        if true_context in found_contexts[:k]:
            accuracy_results[k]["correct"] += 1
            logger.info(f"BM25 Запрос {query_idx}: '{query_text[:50]}...' - Контекст найден в top-{k} ✓")
        else:
            logger.info(f"BM25 Запрос {query_idx}: '{query_text[:50]}...' - Контекст не найден в top-{k} ✗")


def log_final_results(results, top_k_values):
    for k in top_k_values:
        acc = results["accuracy"][k]["accuracy"]
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        logger.info(f"BM25 Точность поиска (top-{k}): {acc:.4f} ({correct}/{total})")

    logger.info(f"BM25 Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")


def finalize_accuracy(accuracy_results):
    for k, data in accuracy_results.items():
        total = data["total"]
        correct = data["correct"]
        data["accuracy"] = correct / total if total > 0 else 0


def calculate_speed_stats(results):
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)
    del results["speed"]["query_times"]


def prepare_sparse_vector(model, text):
    vector = list(model.query_embed(text))[0]
    return {
        "indices": vector.indices.tolist(),
        "values": vector.values.tolist()
    }

def search_bm25(client, collection_name, sparse_vector, limit, search_params):
    start_time = time.time()
    results = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"]
        ),
        using="bm25",
        limit=limit,
        search_params=search_params
    )
    end_time = time.time()
    return results, end_time - start_time

def benchmark_bm25(client, collection_name, test_data, search_params=None, top_k_values=[1, 3]):
    print(f"\n🔍 Запуск оценки производительности BM25 для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности BM25 для коллекции '{collection_name}'")

    results = init_results(top_k_values)
    max_top_k = max(top_k_values)
    total_queries = len(test_data)

    logger.info(f"Оценка производительности BM25 для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска BM25...")

    progress_bar = tqdm(total=total_queries, desc="Обработка запросов BM25", unit="запрос")
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        sparse_vector = prepare_sparse_vector(bm25_embedding_model, query_text)
        search_results, query_time = search_bm25(client, collection_name, sparse_vector, max_top_k, search_params)
        results["speed"]["query_times"].append(query_time)

        found_contexts = [point.payload.get('context', '') for point in search_results.points]

        evaluate_accuracy(results["accuracy"], found_contexts, true_context, top_k_values, query_text, idx)
        progress_bar.update(1)

    progress_bar.close()
    calculate_speed_stats(results)
    finalize_accuracy(results["accuracy"])
    log_final_results(results, top_k_values)

    print(f"✅ Оценка производительности BM25 завершена для коллекции '{collection_name}'")
    return results
