import time
import numpy as np
from tqdm import tqdm
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.models import MultiVectorComparator, Distance, SparseVector, QueryVector

logger = logging.getLogger(__name__)

def benchmark_hybrid_rerank(
    client: QdrantClient,
    collection_name: str,
    test_data,
    search_params=None,
    top_k_values=[1, 3, 5],
    bm25_weight=0.3,
    dense_weight=0.4,
    colbert_weight=0.3,
    reranker=None  # Функция ререйкинга (например, RankGPT, Cohere Rerank и т. д.)
):
    """Бенчмарк производительности Hybrid Search + Rerank в Qdrant"""

    print(f"\n🔍 Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")

    # Результаты
    results = {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {
            "before_rerank": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values},
            "after_rerank": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
        }
    }

    # Максимальный top_k для поиска
    max_top_k = max(top_k_values)

    # Общее количество запросов
    total_queries = len(test_data)
    logger.info(f"Оценка производительности Hybrid Search для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности гибридного поиска...")

    # Создаем прогресс-бар
    progress_bar = tqdm(total=total_queries, desc="Обработка запросов", unit="запрос")

    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    dense_embedding_model = DenseTextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    colbert_embedding_model = LateInteractionEmbedding("colbertv2.0")

    # Обрабатываем каждый запрос
    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        # Генерируем эмбеддинги для запроса
        bm25_vector = list(bm25_embedding_model.query_embed(query_text))[0]
        dense_vector = dense_embedding_model.encode(query_text)
        colbert_vector = colbert_embedding_model.encode(query_text)

        # Подготовка векторов для Qdrant
        query_indices = bm25_vector.indices.tolist()
        query_values = bm25_vector.values.tolist()

        # Измеряем время поиска
        start_time = time.time()

        # Выполняем гибридный поиск
        search_results = client.query_points(
            collection_name=collection_name,
            query={
                "bm25": QueryVector(
                    vector=SparseVector(indices=query_indices, values=query_values),
                    weight=bm25_weight
                ),
                "all-MiniLM-L6-v2": QueryVector(vector=dense_vector, weight=dense_weight),
                "colbertv2.0": QueryVector(vector=colbert_vector, weight=colbert_weight)
            },
            limit=max_top_k,
            search_params=search_params
        )

        end_time = time.time()
        query_time = end_time - start_time
        results["speed"]["query_times"].append(query_time)

        # Список найденных контекстов до ререйкинга
        found_contexts = [point.payload.get('context', '') for point in search_results.points]

        # Оцениваем точность до ререйкинга
        for k in top_k_values:
            results["accuracy"]["before_rerank"][k]["total"] += 1
            if true_context in found_contexts[:k]:
                results["accuracy"]["before_rerank"][k]["correct"] += 1

        # Ререйкинг (если включен)
        if reranker:
            found_contexts = reranker(query_text, found_contexts)

        # Оцениваем точность после ререйкинга
        for k in top_k_values:
            results["accuracy"]["after_rerank"][k]["total"] += 1
            if true_context in found_contexts[:k]:
                results["accuracy"]["after_rerank"][k]["correct"] += 1

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

    # Вычисляем точность до и после ререйкинга
    for stage in ["before_rerank", "after_rerank"]:
        for k in top_k_values:
            correct = results["accuracy"][stage][k]["correct"]
            total = results["accuracy"][stage][k]["total"]
            accuracy = correct / total if total > 0 else 0
            results["accuracy"][stage][k]["accuracy"] = accuracy
            logger.info(f"Hybrid Search {stage.replace('_', ' ')} (top-{k}): {accuracy:.4f} ({correct}/{total})")

    # Выводим статистику скорости
    logger.info(f"Hybrid Search Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(f"Hybrid Search Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")

    print(f"✅ Оценка производительности Hybrid Search завершена для коллекции '{collection_name}'")

    return results