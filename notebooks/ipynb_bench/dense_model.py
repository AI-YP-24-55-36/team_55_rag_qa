import time
from tqdm import tqdm
import numpy as np
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from sentence_transformers import SentenceTransformer
from logger_init import setup_paths, setup_logging
from report_data import (init_results, evaluate_accuracy,
                         calculate_speed_stats, compute_final_accuracy,
                         log_topk_accuracy, log_speed_stats)

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)

# список денз моделей с длинами векторов
MODEL_VECTOR_SIZES = {
    'msmarco-roberta-base-ance-firstp': 768,
    'all-MiniLM-L6-v2': 384,
    'msmarco-MiniLM-L-6-v3': 384,
}


# создание денз коллекции
def create_collection(client, collection_name, vector_size, distance=Distance.COSINE):
    """Создание коллекции в Qdrant"""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        client.delete_collection(collection_name)
        logger.info(f"Коллекция {collection_name} удалена")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "context": VectorParams(size=vector_size, distance=distance)
        }
    )
    logger.info(f"Коллекция {collection_name} создана")


# создание поинтов для загрузки в БД
def build_point_from_memmap(item, idx, vectors):
    vector = vectors[idx].tolist()
    return PointStruct(
        id=item["id"],
        payload=item,
        vector={
            "context": vector
        }
    )


# закгрузка поинтов батчами
def upload_points_in_batches(client, collection_name, points, batch_size=50):
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upload_points(
            collection_name=collection_name,
            points=batch,
        )
        print(f"Загружено {i + len(batch)} из {len(points)} документов")


# чтение эмбеддингов и загрузка в бд
def upload_dense_data(client, collection_name, data, dim, embedding_name: str, batch_size=1,
                      embedding_dir="embeddings", dtype='float32'):
    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с эмбеддингами {embedding_name}")
    start_time = time.time()

    memmap_path = f"{embedding_dir}/{embedding_name}.memmap"
    # чтение векторов
    vectors = np.memmap(memmap_path, dtype=dtype, mode='r').reshape(-1, dim)
    points = []
    progress_bar = tqdm(total=len(data), desc="Подготовка точек", unit="документ")
    # построение поинтов
    for idx, item in enumerate(data):
        point = build_point_from_memmap(item, idx, vectors)
        points.append(point)
        progress_bar.update(1)
    progress_bar.close()
    logger.info(f"🚀 Загрузка {len(points)} точек в Qdrant...")
    upload_points_in_batches(client, collection_name, points, batch_size=batch_size)
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Загрузка завершена за {elapsed_time:.2f} секунд")
    print(f"✅ Загрузка завершена за {elapsed_time:.2f} секунд")


#  создание и загрузка векторов в БД
def upload_dense_model_collections(client, models_to_compare, args, data_for_db):
    for model_name in models_to_compare:
        collection_name = f"{args.collection_name}_{model_name.replace('-', '_')}"
        vector_size = MODEL_VECTOR_SIZES.get(model_name)
        if vector_size is None:
            logger.warning(f"⚠️ Модель {model_name} не найдена ... Пропуск.")
            continue
        logger.info(f"\n📦 Создание коллекции: {collection_name}")
        create_collection(client, collection_name, vector_size)
        logger.info(f"🚀 Загрузка эмбеддингов для модели: {model_name}")
        upload_dense_data(
            client=client,
            collection_name=collection_name,
            data=data_for_db,
            dim=vector_size,
            embedding_name=model_name,
            batch_size=args.batch_size,
            dtype='float32'
        )


# поиск ответа на тестовый запрос
def run_query(client, collection_name, query_vector, search_params, limit):
    start_time = time.time()
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        using="context",
        search_params=search_params,
        limit=limit
    )
    end_time = time.time()
    return search_results, end_time - start_time

# замеры скорости и точности
def benchmark_performance(client, collection_name, test_data, model_name, search_params=None, top_k_values=[1, 3]):
    print(f"\n🔍 Запуск оценки производительности для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности для коллекции '{collection_name}'")
    results = init_results(top_k_values)
    max_top_k = max(top_k_values)
    total_queries = len(test_data)
    model = SentenceTransformer(model_name)
    logger.info(f"Оценка производительности для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска...")
    progress_bar = tqdm(total=total_queries, desc="Обработка запросов", unit="запрос")

    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']
        query_vector = model.encode(query_text)
        search_results, query_time = run_query(client, collection_name, query_vector, search_params, max_top_k)
        results["speed"]["query_times"].append(query_time)
        found_contexts = [point.payload.get('context', '') for point in search_results.points]
        evaluate_accuracy(results["accuracy"], found_contexts, true_context, top_k_values, query_text, idx)
        progress_bar.update(1)

    progress_bar.close()
    calculate_speed_stats(results)
    compute_final_accuracy(results)
    log_topk_accuracy(results, top_k_values)
    log_speed_stats(results)
    print(f"✅ Оценка производительности завершена для коллекции '{collection_name}'")
    return results
