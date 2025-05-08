import datetime
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm
from fastembed import SparseTextEmbedding
from qdrant_client import models

from log_output import Tee
from load_config import load_config

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
OUTPUT_DIR = BASE_DIR / config["paths"]["output_dir"]


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Tee(f"{OUTPUT_DIR}/log_{timestamp}.txt")

logger = logging.getLogger('bench')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(f'{LOGS_DIR}/bench.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)

def benchmark_bm25(client, collection_name, test_data, search_params=None, top_k_values=[1, 3]):
    """Бенчмарк производительности BM25 в Qdrant"""

    print(f"\n🔍 Запуск оценки производительности BM25 для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности BM25 для коллекции '{collection_name}'")

    # Результаты
    results = {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
    }

    # Получаем максимальное значение top_k для поиска
    max_top_k = max(top_k_values)

    # Общее количество запросов
    total_queries = len(test_data)
    logger.info(f"Оценка производительности BM25 для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска BM25...")

    # Создаем прогресс-бар
    progress_bar = tqdm(total=total_queries, desc="Обработка запросов BM25", unit="запрос")

    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    # Обрабатываем каждый запрос отдельно
    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        vector = list(bm25_embedding_model.query_embed(query_text))[0]
        query_indices = vector.indices.tolist()
        query_values = vector.values.tolist()

        # Измеряем время поиска
        start_time = time.time()

        # Выполняем поиск
        search_results = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=query_indices,
                values=query_values,
            ),
            using="bm25",
            limit=max_top_k,
            search_params=search_params
        )

        end_time = time.time()
        query_time = end_time - start_time
        results["speed"]["query_times"].append(query_time)

        found_contexts = []
        for point in search_results.points:
            context = point.payload.get('context', '')
            score = point.score  # Оценка релевантности (если доступна)
            found_contexts.append((context, score))


        for k in top_k_values:
            results["accuracy"][k]["total"] += 1
            if true_context in found_contexts[0][:k]:
                results["accuracy"][k]["correct"] += 1
                logger.info(f"BM25 Запрос {idx}: '{query_text[:50]}...' - Контекст найден в top-{k} ✓")
            else:
                logger.info(f"BM25 Запрос {idx}: '{query_text[:50]}...' - Контекст не найден в top-{k} ✗")

        # Обновляем прогресс-бар
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()

    # Вычисляем статистику скорости
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)

    # Удаляем промежуточные данные о времени запросов
    del results["speed"]["query_times"]

    # Вычисляем точность для каждого значения top_k
    for k in top_k_values:
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        accuracy = correct / total if total > 0 else 0
        results["accuracy"][k]["accuracy"] = accuracy

        logger.info(f"BM25 Точность поиска (top-{k}): {accuracy:.4f} ({correct}/{total})")

    # Выводим статистику скорости
    logger.info(f"BM25 Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(f"BM25 Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")

    print(f"✅ Оценка производительности BM25 завершена для коллекции '{collection_name}'")

    return results

def benchmark_performance(client, collection_name, test_data, model, search_params=None, top_k_values=[1, 3]):
    print(
        f"\n🔍 Запуск оценки производительности для коллекции '{collection_name}'")
    logger.info(
        f"Запуск оценки производительности для коллекции '{collection_name}'")

    # Результаты
    results = {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
    }

    # Получаем максимальное значение top_k для поиска
    max_top_k = max(top_k_values)

    # Общее количество запросов
    total_queries = len(test_data)
    logger.info(f"Оценка производительности для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска...")

    # Создаем прогресс-бар
    progress_bar = tqdm(total=total_queries,
                        desc="Обработка запросов", unit="запрос")

    # Обрабатываем каждый запрос отдельно
    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        # Генерируем эмбеддинг для запроса
        query_vector = model.encode(query_text, show_progress_bar=False)

        # Измеряем время поиска
        start_time = time.time()

        # Выполняем поиск
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            using="context",
            search_params=search_params,
            limit=max_top_k  # Используем максимальное значение top_k
        )

        end_time = time.time()
        query_time = end_time - start_time
        results["speed"]["query_times"].append(query_time)

        # Оцениваем точность для разных значений top_k
        found_contexts = [point.payload.get(
            'context', '') for point in search_results.points]

        # Проверяем точность для каждого значения top_k
        for k in top_k_values:
            # Увеличиваем общее количество запросов
            results["accuracy"][k]["total"] += 1

            # Проверяем, найден ли правильный контекст в первых k результатах
            if true_context in found_contexts[:k]:
                results["accuracy"][k]["correct"] += 1

            else:
                logger.error(
                    f"Запрос {idx}: '{query_text[:50]}...' - Контекст не найден в top-{k} ✗")

        # Обновляем прогресс-бар
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()

    # Вычисляем статистику скорости
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(
            query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)

    # Удаляем промежуточные данные о времени запросов
    del results["speed"]["query_times"]

    # Вычисляем точность для каждого значения top_k
    for k in top_k_values:
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        accuracy = correct / total if total > 0 else 0
        results["accuracy"][k]["accuracy"] = accuracy

        logger.info(
            f"Точность поиска (top-{k}): {accuracy:.4f} ({correct}/{total})")

    # Выводим статистику скорости
    logger.info(
        f"Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(
        f"Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(
        f"Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(
        f"Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")

    print(
        f"✅ Оценка производительности завершена для коллекции '{collection_name}'")

    return results