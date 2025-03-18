import argparse
import os
import logging
import time
from pathlib import Path
from tqdm import tqdm
from qdrant_client.http.models import Distance, SearchParams, HnswConfigDiff
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from read_data_from_csv import read_data
from bench import benchmark_performance, visualize_results, benchmark_tfidf, benchmark_bm25
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import models
from cache_embed import generate_and_save_embeddings
from load_config import load_config
from viz_bm25 import visualize_results_bm25

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]

# Настройка логгера для текущего модуля
logger = logging.getLogger('client')
logger.setLevel(logging.INFO)
logger.propagate = False  # Отключаем передачу логов родительским логгерам

# Создание обработчика для записи логов в файл
file_handler = logging.FileHandler(f'{LOGS_DIR}/client.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description='Бенчмарк для RAG системы')

    # Параметры подключения к Qdrant
    parser.add_argument('--qdrant-host', type=str, default='localhost',
                        help='Хост Qdrant сервера')
    parser.add_argument('--qdrant-port', type=int, default=6333,
                        help='Порт Qdrant сервера')
    parser.add_argument('--collection-name', type=str, default='rag',
                        help='Название коллекции в Qdrant')

    # Параметры модели и поиска
    parser.add_argument('--model-names', nargs='+',
                        default=[
                            'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2', 'TF-IDF'],
                        help='Список моделей для сравнения, включая TF-IDF')
    parser.add_argument('--vector-size', type=int, default=384,
                        help='Размер векторов эмбеддингов')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Размер батча для загрузки данных')
    parser.add_argument('--limit', type=int, default=100,
                        help='Максимальное количество записей для использования')

    # Параметры HNSW
    parser.add_argument('--hnsw-ef', type=int, default=16,
                        help='Параметр ef для HNSW')
    parser.add_argument('--hnsw-m', type=int, default=16,
                        help='Параметр m для HNSW (количество соседей)')
    parser.add_argument('--ef-construct', type=int, default=200,
                        help='Параметр ef_construct для HNSW')

    return parser.parse_args()


def create_collection(client, collection_name, vector_size, distance=Distance.COSINE):
    """Создание коллекции в Qdrant"""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if len(collections):
        for el in collection_names:
            client.delete_collection(el)
            logger.info(f"Коллекция {collection_name} удалена")

    # Создаем коллекцию с именованными векторами
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "context": {
                "size": vector_size,
                "distance": distance
            }
        }
    )
    logger.info(f"Коллекция {collection_name} создана")

def upload_bm25_data(client, collection_name, data):
    """Загрузка данных в Qdrant с использованием встроенного BM25"""

    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием BM25")

    # Проверяем, существует ли коллекция
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if len(collections):
        for el in collection_names:
            client.delete_collection(el)
            logger.info(f"Коллекция {collection_name} удалена")

    # Создаем коллекцию с BM25-индексом

    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF
            )
        }
    )



    logger.info(f"Коллекция {collection_name} создана с поддержкой BM25")



    # Подготавливаем данные для загрузки
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    points = []

    for item in data:

        vector = list(bm25_embedding_model.query_embed(item["context"]))


        if vector:
            sparse_embedding = vector[0]
            points.append(
                models.PointStruct(
                    id=item["id"],
                    payload= item,
                    vector={
                        "bm25": {
                            "values": sparse_embedding.values.tolist(),
                            "indices": sparse_embedding.indices.tolist()
                        }
                    }
                )
            )

    client.upload_points(
        collection_name=collection_name,
        points=points
    )


    logger.info(f"Загрузка данных завершена для коллекции {collection_name}")

def upload_tfidf_data(client, collection_name, data, model):
    """Загрузка данных TF-IDF в Qdrant с использованием разреженных векторов"""
    logger.info(
        f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием TF-IDF")

    # Проверяем, существует ли коллекция
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if len(collections):
        for el in collection_names:
            client.delete_collection(el)
            logger.info(f"Коллекция {collection_name} удалена")


    # Создаем коллекцию для разреженных векторов
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )
    logger.info(
        f"Коллекция {collection_name} создана для разреженных векторов")

    # Извлекаем тексты
    texts = [item["context"] for item in data]

    # Генерируем разреженные векторы

    vectors = model.transform(texts)

    # Загружаем точки
    points = []
    for i in range(vectors.shape[0]):
        indices = vectors[i].indices.tolist()
        values = vectors[i].data.tolist()

        points.append(
            models.PointStruct(
                id=data[i]["id"],
                payload=data[i],
                vector={
                    'text': models.SparseVector(
                        indices=indices, values=values
                    )
                },
            )
        )

    # Загружаем точки в Qdrant
    client.upload_points(
        collection_name=collection_name,
        points=points,
        parallel=4,
        max_retries=3,
    )

    logger.info(
        f"Загрузка данных TF-IDF завершена для коллекции {collection_name}")
def upload_tfidf_data(client, collection_name, data, model):
    """Загрузка данных TF-IDF в Qdrant с использованием разреженных векторов"""
    logger.info(
        f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием TF-IDF")

    # Проверяем, существует ли коллекция
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if len(collections):
        for el in collection_names:
            client.delete_collection(el)
            logger.info(f"Коллекция {collection_name} удалена")

    # Создаем коллекцию для разреженных векторов
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )
    logger.info(
        f"Коллекция {collection_name} создана для разреженных векторов")

    # Извлекаем тексты
    texts = [item["context"] for item in data]

    # Генерируем разреженные векторы

    vectors = model.transform(texts)

    # Загружаем точки
    points = []
    for i in range(vectors.shape[0]):
        indices = vectors[i].indices.tolist()
        values = vectors[i].data.tolist()

        points.append(
            models.PointStruct(
                id=data[i]["id"],
                payload=data[i],
                vector={
                    'text': models.SparseVector(
                        indices=indices, values=values
                    )
                },
            )
        )

    # Загружаем точки в Qdrant
    client.upload_points(
        collection_name=collection_name,
        points=points,
        parallel=4,
        max_retries=3,
    )

    logger.info(
        f"Загрузка данных TF-IDF завершена для коллекции {collection_name}")


def upload_data(client, collection_name, data, model, batch_size=100):
    """Загрузка данных в Qdrant"""
    logger.info(
        f"Загрузка {len(data)} документов в коллекцию {collection_name}")
    start_time = time.time()

    # Создаем прогресс-бар
    progress_bar = tqdm(
        total=len(data), desc="Загрузка данных", unit="документ")

    # Загрузка данных батчами
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        texts = [item["context"] for item in batch]

        # Генерация эмбеддингов для батча
        args = parse_args()
        vectors = generate_and_save_embeddings(
            texts=texts,
            model=model,
            array_name=f'{collection_name}+{args.limit}',
            save_dir="embeddings"
        )


        # Подготовка точек для загрузки
        points = []
        for j, (item, vector) in enumerate(zip(batch, vectors)):
            points.append({
                "id": item["id"],
                "vector": {
                    "context": vector.tolist()
                },
                "payload": item
            })

        # Загрузка в Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        # Обновляем прогресс-бар
        progress_bar.update(len(batch))

        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(data):
            logger.info(
                f"Загружено {min(i + batch_size, len(data))}/{len(data)} документов")

    # Закрываем прогресс-бар
    progress_bar.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Загрузка данных завершена за {elapsed_time:.2f} секунд")
    print(f"✅ Загрузка данных завершена за {elapsed_time:.2f} секунд")


def main():
    args = parse_args()

    # Уведомление о запуске бенчмарка
    print("\n" + "="*80)
    print("🚀 ЗАПУСК БЕНЧМАРКА RAG СИСТЕМЫ")
    print("="*80)
    logger.info("Запуск бенчмарка RAG системы")

    # Инициализация клиента Qdrant
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # Загрузка данных с ограничением по размеру
    logger.info(f"Загрузка данных с limit={args.limit}")
    print(f"📂 Загрузка данных (limit={args.limit})...")
    data_for_db, data_df = read_data(limit=args.limit)

    logger.info(f"Загружено {len(data_for_db)} документов")
    print(f"✅ Загружено {len(data_for_db)} документов")

    # Получаем список моделей из аргументов командной строки
    all_models = args.model_names
    logger.info(f"Выбранные модели для сравнения: {', '.join(all_models)}")
    print(f"🔄 Выбранные модели для сравнения: {', '.join(all_models)}")

    # Разделяем модели на обычные и TF-IDF
    # models_to_compare = [model for model in all_models if model != 'BM25']
    models_to_compare = [model for model in all_models if model not in ['BM25', 'TF-IDF']]
    use_tfidf = 'TF-IDF' in all_models
    use_bm25 = 'BM25' in all_models

    # Инициализация моделей
    model_instances = {}
    if models_to_compare:
        print(f"🔄 Инициализация моделей для dense векторов...")
        progress_bar = tqdm(total=len(models_to_compare),
                            desc="Загрузка моделей", unit="модель")

        for model_name in models_to_compare.copy():
            try:
                logger.info(f"Инициализация модели: {model_name}")
                # Инициализируем модель SentenceTransformer
                model_instances[model_name] = SentenceTransformer(model_name)
                logger.info(f"Модель {model_name} инициализирована")
                progress_bar.update(1)
            except Exception as e:
                logger.error(
                    f"Ошибка при инициализации модели {model_name}: {e}")
                models_to_compare.remove(model_name)

        progress_bar.close()
        print(f"✅ Модели инициализированы: {', '.join(models_to_compare)}")

    # Инициализация TF-IDF модели отдельно, если она выбрана
    tfidf_model = None
    if use_tfidf:
        print(f"🔄 Инициализация модели TF-IDF...")
        logger.info("Инициализация модели TF-IDF")

        # Извлекаем тексты для обучения TF-IDF
        corpus_texts = [item["context"] for item in data_for_db]

        # Создаем и обучаем модель TF-IDF
        tfidf_model = TfidfVectorizer(stop_words='english', ngram_range=(
            1, 2), max_df=0.85, sublinear_tf=True)
        tfidf_model.fit(corpus_texts)
        logger.info("Модель TF-IDF инициализирована")
        print(f"✅ Модель TF-IDF инициализирована")

    # Определение алгоритмов поиска для dense векторов
    search_algorithms = {
        "Exact Search": SearchParams(exact=True),
        f"HNSW Users ef={args.hnsw_ef}": SearchParams(hnsw_ef=args.hnsw_ef),
        "HNSW High Precision ef=512": SearchParams(hnsw_ef=512)
    }

    # Результаты бенчмарка для dense векторов
    speed_results = {}
    accuracy_results = {}

    # Запуск бенчмарка для каждой модели с dense векторами
    if models_to_compare:
        for model_name in models_to_compare:
            model = model_instances[model_name]

            # Создание коллекции
            collection_name = f"{args.collection_name}_{model_name.replace('-', '_')}"
            create_collection(client, collection_name, args.vector_size)

            # Загрузка данных
            upload_data(client, collection_name,
                        data_for_db, model, args.batch_size)

            # Результаты для текущей модели
            speed_results[model_name] = {}
            accuracy_results[model_name] = {}

            # Оценка для каждого алгоритма
            for algo_name, search_params in search_algorithms.items():
                logger.info(
                    f"Оценка алгоритма {algo_name} с моделью {model_name}")
                print(
                    f"\n🔍 Оценка алгоритма {algo_name} с моделью {model_name}")

                # Обновление параметров HNSW
                if algo_name.startswith("HNSW"):
                    client.update_collection(
                        collection_name=collection_name,
                        hnsw_config=HnswConfigDiff(
                            m=args.hnsw_m,
                            ef_construct=args.ef_construct,
                        )
                    )

                # Запуск объединенной функции оценки производительности
                benchmark_results = benchmark_performance(
                    client=client,
                    collection_name=collection_name,
                    test_data=data_df,
                    model=model,
                    search_params=search_params,
                    top_k_values=[1, 3]
                )

                # Сохраняем результаты скорости
                speed_results[model_name][algo_name] = benchmark_results["speed"]

                # Сохраняем результаты точности
                accuracy_results[model_name][algo_name] = benchmark_results["accuracy"]
    # Запуск бенчмарка для BM25, если он выбран
    bm25_results = None
    if use_bm25:
        print("\n" + "=" * 80)
        print("🔍 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ BM25")
        print("=" * 80)
        logger.info("Запуск оценки производительности BM25")

        # Создание коллекции для BM25
        bm25_collection_name = f"{args.collection_name}_bm25"

        # Загрузка данных BM25
        upload_bm25_data(client, bm25_collection_name, data_for_db)

        # Результаты для BM25
        bm25_speed_results = {}
        bm25_accuracy_results = {}

        # Оценка для каждого алгоритма поиска
        for algo_name, search_params in search_algorithms.items():
            logger.info(f"Оценка алгоритма {algo_name} с моделью BM25")
            print(f"\n🔍 Оценка алгоритма {algo_name} с моделью BM25")

            # Запуск бенчмарка для BM25 с текущими параметрами поиска
            benchmark_results = benchmark_bm25(
                client=client,
                collection_name=bm25_collection_name,
                test_data=data_df,
                search_params=search_params,
                top_k_values=[1, 3]
            )

            # Сохраняем результаты скорости
            bm25_speed_results[algo_name] = benchmark_results["speed"]

            # Сохраняем результаты точности
            bm25_accuracy_results[algo_name] = benchmark_results["accuracy"]

        # Сохраняем результаты BM25 для визуализации
        bm25_results = {
            "speed": bm25_speed_results,
            "accuracy": bm25_accuracy_results
        }

        # Вывод результатов скорости
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ СКОРОСТИ ПОИСКА")
        print("=" * 80)

    # Запуск бенчмарка для TF-IDF, если она выбрана
    tfidf_results = None
    if use_tfidf and tfidf_model:
        print("\n" + "="*80)
        print("🔍 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ TF-IDF")
        print("="*80)
        logger.info("Запуск оценки производительности TF-IDF")

        # Создание коллекции для TF-IDF
        tfidf_collection_name = f"{args.collection_name}_tfidf"

        # Загрузка данных TF-IDF
        upload_tfidf_data(client, tfidf_collection_name,
                          data_for_db, tfidf_model)

        # Результаты для TF-IDF
        tfidf_speed_results = {}
        tfidf_accuracy_results = {}

        # Оценка для каждого алгоритма поиска
        for algo_name, search_params in search_algorithms.items():
            logger.info(f"Оценка алгоритма {algo_name} с моделью TF-IDF")
            print(f"\n🔍 Оценка алгоритма {algo_name} с моделью TF-IDF")

            # Запуск бенчмарка для TF-IDF с текущими параметрами поиска
            benchmark_results = benchmark_tfidf(
                client=client,
                collection_name=tfidf_collection_name,
                test_data=data_df,
                model=tfidf_model,
                search_params=search_params,
                top_k_values=[1, 3]
            )

            # Сохраняем результаты скорости
            tfidf_speed_results[algo_name] = benchmark_results["speed"]

            # Сохраняем результаты точности
            tfidf_accuracy_results[algo_name] = benchmark_results["accuracy"]

        # Сохраняем результаты TF-IDF для визуализации
        tfidf_results = {
            "speed": tfidf_speed_results,
            "accuracy": tfidf_accuracy_results
        }

        # Вывод результатов скорости
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ СКОРОСТИ ПОИСКА")
        print("="*80)

    # Запуск бенчмарка для каждой модели с dense векторами
    if models_to_compare:
        for model_name in models_to_compare:
            model = model_instances[model_name]

            # Создание коллекции
            collection_name = f"{args.collection_name}_{model_name.replace('-', '_')}"
            create_collection(client, collection_name, args.vector_size)

            # Загрузка данных
            upload_data(client, collection_name,
                        data_for_db, model, args.batch_size)

            # Результаты для текущей модели
            speed_results[model_name] = {}
            accuracy_results[model_name] = {}

            # Оценка для каждого алгоритма
            for algo_name, search_params in search_algorithms.items():
                logger.info(
                    f"Оценка алгоритма {algo_name} с моделью {model_name}")
                print(
                    f"\n🔍 Оценка алгоритма {algo_name} с моделью {model_name}")

                # Обновление параметров HNSW
                if algo_name.startswith("HNSW"):
                    client.update_collection(
                        collection_name=collection_name,
                        hnsw_config=HnswConfigDiff(
                            m=args.hnsw_m,
                            ef_construct=args.ef_construct,
                        )
                    )

                # Запуск объединенной функции оценки производительности
                benchmark_results = benchmark_performance(
                    client=client,
                    collection_name=collection_name,
                    test_data=data_df,
                    model=model,
                    search_params=search_params,
                    top_k_values=[1, 3]
                )

                # Сохраняем результаты скорости
                speed_results[model_name][algo_name] = benchmark_results["speed"]

                # Сохраняем результаты точности
                accuracy_results[model_name][algo_name] = benchmark_results["accuracy"]

    # Запуск бенчмарка для TF-IDF, если она выбрана
    tfidf_results = None
    if use_tfidf and tfidf_model:
        print("\n" + "=" * 80)
        print("🔍 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ TF-IDF")
        print("=" * 80)
        logger.info("Запуск оценки производительности TF-IDF")

        # Создание коллекции для TF-IDF
        tfidf_collection_name = f"{args.collection_name}_tfidf"

        # Загрузка данных TF-IDF
        upload_tfidf_data(client, tfidf_collection_name,
                          data_for_db, tfidf_model)

        # Результаты для TF-IDF
        tfidf_speed_results = {}
        tfidf_accuracy_results = {}

        # Оценка для каждого алгоритма поиска
        for algo_name, search_params in search_algorithms.items():
            logger.info(f"Оценка алгоритма {algo_name} с моделью TF-IDF")
            print(f"\n🔍 Оценка алгоритма {algo_name} с моделью TF-IDF")

            # Запуск бенчмарка для TF-IDF с текущими параметрами поиска
            benchmark_results = benchmark_tfidf(
                client=client,
                collection_name=tfidf_collection_name,
                test_data=data_df,
                model=tfidf_model,
                search_params=search_params,
                top_k_values=[1, 3]
            )

            # Сохраняем результаты скорости
            tfidf_speed_results[algo_name] = benchmark_results["speed"]

            # Сохраняем результаты точности
            tfidf_accuracy_results[algo_name] = benchmark_results["accuracy"]

        # Сохраняем результаты TF-IDF для визуализации
        tfidf_results = {
            "speed": tfidf_speed_results,
            "accuracy": tfidf_accuracy_results
        }

        # Вывод результатов скорости
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ СКОРОСТИ ПОИСКА")
        print("=" * 80)

    # Вывод результатов для dense векторов
    if models_to_compare:
        for model_name in models_to_compare:
            print(f"\nМодель: {model_name}")

            for algo_name in speed_results[model_name].keys():
                result = speed_results[model_name][algo_name]

                print(f"  Алгоритм: {algo_name}")
                print(f"    Среднее время: {result['avg_time'] * 1000:.2f} мс")
                print(
                    f"    Медианное время: {result['median_time'] * 1000:.2f} мс")
                print(
                    f"    Максимальное время: {result['max_time'] * 1000:.2f} мс")
                print(
                    f"    Минимальное время: {result['min_time'] * 1000:.2f} мс")

    # Вывод результатов для TF-IDF
    if use_tfidf and tfidf_results:
        print(f"\nМодель: TF-IDF")

        for algo_name in tfidf_results["accuracy"].keys():
            print(f"  Алгоритм: {algo_name}")

            for k in [1, 3]:
                if k in tfidf_results["accuracy"][algo_name]:
                    result = tfidf_results["accuracy"][algo_name][k]
                    print(
                        f"    Top-{k}: Точность = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

        # Вывод результатов точности
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ ТОЧНОСТИ ПОИСКА")
        print("=" * 80)

    # Вывод результатов для dense векторов
    if models_to_compare:
        for model_name in models_to_compare:
            print(f"\nМодель: {model_name}")

            for algo_name in accuracy_results[model_name].keys():
                print(f"  Алгоритм: {algo_name}")

                for k in [1, 3]:
                    if k in accuracy_results[model_name][algo_name]:
                        result = accuracy_results[model_name][algo_name][k]
                        print(
                            f"    Top-{k}: Точность = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Визуализация результатов
    if (models_to_compare or use_tfidf):
        visualize_results(
            speed_results=speed_results,
            accuracy_results=accuracy_results,
            tfidf_results=tfidf_results,
            title_prefix="Сравнение производительности RAG системы",
            save_dir=f"{GRAPHS_DIR}"
        )

        logger.info("Бенчмарк завершен успешно")
        print("\n" + "=" * 80)
        print("✅ БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО")
        print(f"Графики сохранены в директории {GRAPHS_DIR}")
        print("=" * 80)

    # Вывод результатов для BM25
    if use_bm25 and bm25_results:
        print(f"\nМодель: BM25")

        for algo_name, speed_data in bm25_results["speed"].items():
            print(f"  Алгоритм: {algo_name}")
            print(f"    Среднее время: {speed_data['avg_time'] * 1000:.2f} мс")
            print(f"    Медианное время: {speed_data['median_time'] * 1000:.2f} мс")
            print(f"    Максимальное время: {speed_data['max_time'] * 1000:.2f} мс")
            print(f"    Минимальное время: {speed_data['min_time'] * 1000:.2f} мс")

        # Вывод результатов точности
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ ТОЧНОСТИ ПОИСКА")
        print("=" * 80)

    # Вывод результатов для dense векторов
    if models_to_compare:
        for model_name in models_to_compare:
            print(f"\nМодель: {model_name}")

            for algo_name in accuracy_results[model_name].keys():
                print(f"  Алгоритм: {algo_name}")

                for k in [1, 3]:
                    if k in accuracy_results[model_name][algo_name]:
                        result = accuracy_results[model_name][algo_name][k]
                        print(
                            f"    Top-{k}: Точность = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Вывод результатов для BM25
    if use_bm25 and bm25_results:
        print(f"\nМодель: BM25")

        for algo_name, accuracy_data in bm25_results["accuracy"].items():
            print(f"  Алгоритм: {algo_name}")

            for k in [1, 3]:
                if k in accuracy_data:
                    result = accuracy_data[k]
                    print(f"    Top-{k}: Точность = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Визуализация результатов
    if (models_to_compare or use_bm25):
        visualize_results_bm25(
            speed_results=speed_results,
            accuracy_results=accuracy_results,
            bm25_results=bm25_results,
            title_prefix="Сравнение производительности RAG системы",
            save_dir=f"{GRAPHS_DIR}"
        )
    #
    logger.info("Бенчмарк завершен успешно")
    print("\n" + "=" * 80)
    print("✅ БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО")
    print(f"Графики сохранены в директории {GRAPHS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()