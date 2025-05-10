import time
from tqdm import tqdm
import pickle
from fastembed import SparseTextEmbedding
from qdrant_client import models
from logger_init import setup_paths, setup_logging
from report_data import (init_results, evaluate_accuracy,
                         calculate_speed_stats, compute_final_accuracy,
                         log_topk_accuracy, log_speed_stats)

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)


# функция создания коллекции
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


#  загрузка данных в бд с использованием предрассчитанных BM25 эмбеддингов.
def upload_bm25_data(client, collection_name, data):
    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием BM25 (из файла)")
    create_coll(client, collection_name)

    # загрузка эмбеддингов из файла
    with open('embeddings/sparse_embeddings.pkl', 'rb') as f:
        sparse_embeddings = pickle.load(f)

    points = []
    for item, sparse_embedding in zip(data, sparse_embeddings):
        if sparse_embedding is not None:
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

    logger.info(f"✅ Загрузка данных завершена для коллекции {collection_name}")


# функция для подготовки эмбеддинга для тестового запроса
def prepare_sparse_vector(model, text):
    vector = list(model.query_embed(text))[0]
    return {
        "indices": vector.indices.tolist(),
        "values": vector.values.tolist()
    }


# поиск по БД и извлечение
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


# замеры скорости и точности
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
    compute_final_accuracy(results)
    log_topk_accuracy(results, top_k_values)
    log_speed_stats(results)

    print(f"✅ Оценка производительности BM25 завершена для коллекции '{collection_name}'")
    return results

# запуск бенчмарка
def run_benchmark_bm25_model(client, base_collection_name, data_for_db, data_df, search_algorithms):
    print("\n" + "=" * 80)
    print("🔍 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ BM25")
    print("=" * 80)
    logger.info("Запуск оценки производительности BM25")
    bm25_collection_name = f"{base_collection_name}_bm25"
    upload_bm25_data(client, bm25_collection_name, data_for_db)
    bm25_speed_results = {}
    bm25_accuracy_results = {}
    for algo_name, search_params in search_algorithms.items():
        logger.info(f"Оценка алгоритма {algo_name} с моделью BM25")
        print(f"\n🔍 Оценка алгоритма {algo_name} с моделью BM25")
        benchmark_results = benchmark_bm25(
            client=client,
            collection_name=bm25_collection_name,
            test_data=data_df,
            search_params=search_params,
            top_k_values=[1, 3]
        )
        bm25_speed_results[algo_name] = benchmark_results["speed"]
        bm25_accuracy_results[algo_name] = benchmark_results["accuracy"]
    return {
        "speed": bm25_speed_results,
        "accuracy": bm25_accuracy_results
    }
