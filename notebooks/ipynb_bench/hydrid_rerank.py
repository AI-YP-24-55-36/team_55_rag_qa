import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    Modifier,
    MultiVectorConfig,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from client import parse_args
from load_config import load_config
from read_data_from_csv import read_data

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

def upload_hybrid_data(client, collection_name: str, data):
    """Загрузка данных в Qdrant с поддержкой гибридного поиска (BM25 + Dense + ColBERT)"""

    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с гибридным поиском")

    # Проверяем, существует ли коллекция
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if len(collections):
        for el in collection_names:
            client.delete_collection(el)
            print(f"Collection {el} has been cleared")

    # Создаем коллекцию с гибридным индексом
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=384,  # Размерность MiniLM
                distance=Distance.COSINE
            ),
            "colbertv2.0": VectorParams(
                size=128,  # Размерность ColBERT
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

    logger.info(f"Коллекция {collection_name} создана с поддержкой гибридного поиска")
    print(f"Коллекция {collection_name} создана с поддержкой гибридного поиска")

    # Загружаем модели эмбеддингов
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    dense_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    colbert_embedding_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")

    # Переводим модель в режим оценки и на CPU
    colbert_embedding_model.eval()
    colbert_embedding_model = colbert_embedding_model.to('cpu')

    # Подготавливаем данные для загрузки
    points = []

    try:
        for item in tqdm(data):
            text = item["context"]

            try:
                # BM25 вектор
                sparse_vector = list(bm25_embedding_model.query_embed(text))
                sparse_embedding = sparse_vector[0] if sparse_vector else None

                # Dense вектор (MiniLM)
                dense_embedding = dense_embedding_model.encode(text)
                dense_embedding = dense_embedding.tolist()

                # ColBERT вектор
                inputs = colbert_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512  # Ограничиваем длину входного текста
                )

                with torch.no_grad():
                    outputs = colbert_embedding_model(**inputs)
                    # Берем среднее по токенам и приводим к нужной размерности
                    colbert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    # Убеждаемся, что размерность правильная
                    if len(colbert_embedding.shape) > 1:
                        colbert_embedding = colbert_embedding.mean(dim=0)
                    # Приводим к нужной размерности если нужно
                    if colbert_embedding.shape[0] != 128:
                        colbert_embedding = torch.nn.functional.interpolate(
                            colbert_embedding.unsqueeze(0).unsqueeze(0),
                            size=128,
                            mode='linear'
                        ).squeeze()

                    colbert_embedding = colbert_embedding.cpu().numpy().tolist()

                # Создаем точку данных
                point = models.PointStruct(
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
                points.append(point)

            except Exception as e:
                logger.error(f"Ошибка при обработке документа {item['id']}: {str(e)}")
                continue

        print(f"Создано {len(points)} points")

        # Загружаем данные в Qdrant батчами
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upload_points(
                collection_name=collection_name,
                points=batch
            )
            print(f"Загружено {i + len(batch)} из {len(points)} документов")

        logger.info(f"✅ Данные успешно загружены в коллекцию {collection_name}")
        print(f"✅ Данные успешно загружены в коллекцию {collection_name}")

    except Exception as e:
        logger.error(f"Произошла ошибка при загрузке данных: {str(e)}")
        raise


def benchmark_hybrid_rerank(client, collection_name, test_data, model, search_params=None, top_k_values=[1, 3],
                            bm25_weight=0.3,
                            dense_weight=0.4,
                            colbert_weight=0.3,
                            reranker = None
                            ):
    """Бенчмарк производительности Hybrid Search + Rerank в Qdrant с использованием prefetch"""

    print(f"\n🔍 Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности Гибридного Поиска + Реранка для коллекции '{collection_name}'")

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

    max_top_k = max(top_k_values)
    total_queries = len(test_data)

    logger.info(f"Оценка производительности Hybrid Search для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности гибридного поиска...")

    progress_bar = tqdm(total=total_queries, desc="Обработка запросов", unit="запрос")

    # Загружаем модели эмбеддингов
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    dense_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    colbert_embedding_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")

    # Переводим ColBERT в режим оценки и на CPU
    colbert_embedding_model.eval()
    colbert_embedding_model = colbert_embedding_model.to('cpu')

    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']


        # BM25 вектор
        sparse_vector = list(bm25_embedding_model.query_embed(query_text))
        sparse_embedding = sparse_vector[0] if sparse_vector else None

        # Dense вектор (MiniLM)
        dense_embedding = dense_embedding_model.encode(query_text)
        dense_embedding = dense_embedding.tolist()

        # ColBERT вектор
        inputs = colbert_tokenizer(
            query_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = colbert_embedding_model(**inputs)
            colbert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            if len(colbert_embedding.shape) > 1:
                colbert_embedding = colbert_embedding.mean(dim=0)
            if colbert_embedding.shape[0] != 128:
                colbert_embedding = torch.nn.functional.interpolate(
                    colbert_embedding.unsqueeze(0).unsqueeze(0),
                    size=128,
                    mode='linear'
                ).squeeze()
            colbert_embedding = colbert_embedding.cpu().numpy().tolist()

        # Измеряем время поиска
        start_time = time.time()

        # Создаем prefetch запросы
        prefetch = [
            models.Prefetch(
                query=dense_embedding,
                using="dense",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_embedding.indices.tolist() if sparse_embedding else [],
                    values=sparse_embedding.values.tolist() if sparse_embedding else []
                ),
                using="bm25",
                limit=20,
            ),
        ]

        # Выполняем поиск
        search_results = client.query_points(
            collection_name,
            prefetch=prefetch,
            query=colbert_embedding,
            using="colbertv2.0",
            with_payload=True,
            limit=max_top_k,
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
            results["accuracy"]["before_rerank"][k]["total"] += 1
            if true_context in found_contexts[0][:k]:
                results["accuracy"]["before_rerank"][k]["correct"] += 1

        # Ререйкинг (если включен)
        if reranker:
            found_contexts_r = reranker(query_text, found_contexts)

    # Оцениваем точность после ререйкинга
            for k in top_k_values:
                results["accuracy"]["after_rerank"][k]["total"] += 1
                if true_context in found_contexts_r[:k]:
                    results["accuracy"]["after_rerank"][k]["correct"] += 1


        progress_bar.update(1)

    progress_bar.close()

    # Вычисляем статистику скорости
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)
    del results["speed"]["query_times"]

    # Вычисляем точность
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

def simple_reranker(query, candidates):

    texts = [(query, context) for context, _ in candidates]
    scores = reranker_model.predict(texts)

    # Добавляем новые оценки и сортируем по ним
    reranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Возвращаем только контексты в новом порядке
    return [context for (context, _), _ in reranked_results]


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
                             title_prefix="Сравнение реранкинга", save_dir=f"{GRAPHS_DIR}/graphs"):

    print(f"\n📊 Создание визуализаций результатов реранкинга...")
    logger.info("Создание визуализаций результатов реранкинга")

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

    # Создаём график времени
    plt.bar(["Без реранкинга", "С реранкингом"], speeds, color=['#1f77b4', '#2ca02c'])  # Синие и зелёные цвета
    plt.ylabel("Время (мс)")
    plt.title(f"{title_prefix}: Время выполнения запроса")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Сохранение графика (до plt.show())
    speed_save_path = f"{save_dir}/speed_comparison_{timestr}_hybrid.png"
    plt.savefig(speed_save_path, dpi=300, bbox_inches='tight')
    logger.info(f"График скорости сохранен в {speed_save_path}")
    print(f"График скорости сохранен в {speed_save_path}")

    # --- 2️⃣ Визуализация точности поиска ---
    plt.figure(figsize=(10, 5))

    acc_before = [results_without_rerank["accuracy"]["before_rerank"][k]["accuracy"] for k in top_k_values]
    acc_after = [results_with_rerank["accuracy"]["after_rerank"][k]["accuracy"] for k in top_k_values]

    x = np.arange(len(top_k_values))  # позиции по X
    width = 0.3  # ширина столбцов

    # Создаём график точности
    plt.bar(x - width / 2, acc_before, width, label="Без реранкинга", color='#1f77b4')  # Синий цвет
    plt.bar(x + width / 2, acc_after, width, label="С реранкингом", color='#ff7f0e')  # Оранжевый цвет

    plt.xticks(ticks=x, labels=[f"Top-{k}" for k in top_k_values])  # подписи оси X
    plt.ylabel("Точность (Accuracy)")
    plt.title(f"{title_prefix}: Точность поиска")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Сохранение графика (до plt.show())
    accuracy_save_path = f"{save_dir}/accuracy_comparison_{timestr}_hybrid.png"
    plt.savefig(accuracy_save_path, dpi=300, bbox_inches='tight')
    logger.info(f"График точности сохранен в {accuracy_save_path}")
    print(f"График точности сохранен в {accuracy_save_path}")

if __name__ == "__main__":

    args = parse_args()
    args.limit = 1000

    data_for_db, data_df = read_data(limit=args.limit)
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

        # Загрузка данных
    upload_hybrid_data(
        client=client,
        collection_name="hybrid_collection",
        data=data_for_db
    )

    all_models = args.model_names
    results_without_rerank = benchmark_hybrid_rerank(
        client=client,
        collection_name="hybrid_collection",
        test_data=data_df,
        model=all_models[0],
        search_params={"exact": False, "hnsw_ef": 128},
        top_k_values=[1, 3, 5],
        bm25_weight=0.3,
        dense_weight=0.4,
        colbert_weight=0.3,
        reranker=None
    )


    # Загружаем модель для реранкинга
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Запускаем бенчмарк с реранкингом
    results_with_rerank = benchmark_hybrid_rerank(
        client=client,
        collection_name="hybrid_collection",
        test_data=data_df,
        model=None,
        reranker=simple_reranker  # Передаем функцию реранкинга
    )


    print_comparison(results_without_rerank, results_with_rerank)
    visualize_results_rerank(results_without_rerank, results_with_rerank)
