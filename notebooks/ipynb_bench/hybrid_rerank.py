import time
import pickle
import numpy as np
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding, TextEmbedding
# from sentence_transformers import CrossEncoder
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models
from qdrant_client.models import (
    Distance,
    Modifier,
    OptimizersConfigDiff,
    HnswConfigDiff,
    MultiVectorConfig,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    PointStruct
)

from tqdm import tqdm
from logger_init import setup_paths, setup_logging


BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)

# модель для реранкинга
# reranker_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2") - самая быстрая
reranker_model = TextCrossEncoder(model_name='jinaai/jina-reranker-v1-turbo-en')
# BAAI/bge-reranker-base - очень медленная
# jinaai/jina-reranker-v1-turbo-en - средняя скорость

# функция удаляет коллекцию с таким именем, если она существует
def clear_existing_collections(client,  collection_name):
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        print(f"Колекция {collection_name} существует, удаляем" )
        logger.info(f"Колекция {collection_name} существует, удаляем" )

        client.delete_collection(collection_name)
        logger.info(f"Коллекция {collection_name} удалена")

# функция для создания коллекции
def create_hybrid_collection(client, collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=1024,
                distance=Distance.COSINE
            ),

            "colbertv2.0": VectorParams(
                size=128,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator="max_sim"),
                hnsw_config=HnswConfigDiff(
                    m=0  # отключение построение графа HNSW
                )
            ),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=False,

            ),
                modifier=Modifier.IDF,
                )

        },
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0
        )

    )
    logger.info(f"Создана коллекция {collection_name}, готова к заполнению")
    print(f"Создана коллекция {collection_name}, готова к заполнению")

dense_embeddings = np.memmap('embeddings/mxbai-embed-large-v1.memmap', dtype='float32', mode='r').reshape(-1, 1024)
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
    # загрузка эмбеддингов по индексу из файлов memmap
    sparse_embedding = sparse_embeddings[idx]
    dense_embedding = dense_embeddings[idx].tolist()
    colbert_embedding = colbert_embeddings[idx].tolist()

    return PointStruct(
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
    clear_existing_collections(client, collection_name)
    create_hybrid_collection(client, collection_name)
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

    # запуск индексации
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=5000),
    )




#  фунция загрузки моделей для создания эмбеддингов тестовых запросов
def load_embedding():
    return {
        "bm25": SparseTextEmbedding("Qdrant/bm25"),
        "dense": TextEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1"),
        "colbert": LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    }
# кодировка тестового запроса
def encode_query(query_text, models):
    sparse_vector = list(models["bm25"].query_embed(query_text))
    sparse_embedding = sparse_vector[0] if sparse_vector else None
    # dense_embedding = models["dense"].encode_corpus([{"text": query_text}], convert_to_tensor=False)[0]
    dense_embedding = list(models["dense"].embed(query_text, normalize=True))[0]
    colbert_embedding = list(models["colbert"].embed(query_text))[0]
    return sparse_embedding, dense_embedding, colbert_embedding

# поиск контекста в БД
def run_hybrid_search(client, collection_name, sparse_embedding, dense_embedding, colbert_embedding, top_k):
    prefetch = [
        models.Prefetch(query=dense_embedding, using="dense", limit=top_k),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_embedding.indices.tolist() if sparse_embedding else [],
                values=sparse_embedding.values.tolist() if sparse_embedding else []
            ),
            using="bm25",
            limit=top_k,
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

# запуск бенчмарка
def benchmark_hybrid_rerank(client, collection_name, test_data, top_k_values, reranker=None):
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

# функция под TextCrossEncoder
def reranker(query, candidates, top_k=None):
    #  пары (query, context)
    texts = [context for context, _ in candidates]

    #  оценки от модели через .rerank()
    new_scores = list(reranker_model.rerank(query, texts))

    # сопоставление индексы и оценки
    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)

    # сортировка кандидатов по оценкам
    reranked = [(texts[i], score) for i, score in ranking]

    if top_k is not None:
        return reranked[:top_k]
    return reranked

#
# # функция под CrossEncoder
# def reranker(query, candidates, top_k=None):
#     texts = [(query, context) for context, _ in candidates]
#     scores = reranker_model.predict(texts)
#     # Добавляем новые оценки и сортируем по ним
#     reranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#     # Возвращаем кортежи: (context, new_score)
#     return [(context, score) for (context, _), score in reranked_results]


def print_comparison(results_without_rerank, results_with_rerank, top_k_values):
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


def run_bench_hybrid(client, data_for_db, data_df, load, top_k_values):
    if load == 1:
        # Загрузка данных
        upload_hybrid_data(
            client=client,
            collection_name="hybrid_collection",
            data=data_for_db
        )
    else:
        logger.info(f"🔍 Не загружаем данные, параметр load=0")
        print(f"\n🔍Не загружаем данные, параметр load=0")

    results_without_rerank = benchmark_hybrid_rerank(
        client=client,
        collection_name="hybrid_collection",
        test_data=data_df,
        top_k_values=top_k_values,
        reranker=None
    )

    # Запускаем бенчмарк с реранкингом
    results_with_rerank = benchmark_hybrid_rerank(
        client=client,
        collection_name="hybrid_collection",
        test_data=data_df,
        top_k_values=top_k_values,
        reranker=reranker  # передача функцию реранкинга
    )

    return results_without_rerank, results_with_rerank