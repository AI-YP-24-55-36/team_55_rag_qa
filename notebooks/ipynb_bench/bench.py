import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams, HnswConfigDiff
from read_data_from_csv import read_data
from logger_init import setup_paths, setup_logging
from visualisation import visualize_results
from hybrid_rerank import print_comparison, run_bench_hybrid
from visualisation import visualize_results_rerank
from sparse_bm25 import upload_bm25_data, run_benchmark_bm25_model
from report_data import print_speed_results, print_accuracy_results
from dense_model import upload_dense_model_collections, benchmark_performance

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)


# aeфункция для инициализации аргументов запуска бенчмарка
def parse_args():
    parser = argparse.ArgumentParser(description='Бенчмарк для RAG системы')
    # параметры подключения к Qdrant
    parser.add_argument('--qdrant-host', type=str, default='localhost',
                        help='Хост Qdrant сервера')
    parser.add_argument('--qdrant-port', type=int, default=6333,
                        help='Порт Qdrant сервера')
    parser.add_argument('--collection-name', type=str, default='rag',
                        help='Название коллекции в Qdrant')

    # список моделей
    parser.add_argument('--model-names', nargs='+',
                        default=[
                            "all-MiniLM-L6-v2",
                            "msmarco-MiniLM-L-6-v3",
                            "msmarco-roberta-base-ance-firstp",
                            'BM25'],

                        help='Список моделей для сравнения, включая BM25')
    parser.add_argument('--vector-size', type=int, default=384,
                        help='Размер векторов эмбеддингов')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Размер батча для загрузки данных')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Максимальное количество записей для использования')

    # параметры HNSW для dense моделей
    parser.add_argument('--hnsw-ef', type=int, default=16,
                        help='Параметр ef для HNSW')
    parser.add_argument('--hnsw-m', type=int, default=16,
                        help='Параметр m для HNSW (количество соседей)')
    parser.add_argument('--ef-construct', type=int, default=200,
                        help='Параметр ef_construct для HNSW')

    parser.add_argument('--hybrid', type=int, default=0,
                        help='Параметр для запуска гибридного поиска')

    return parser.parse_args()


def initialize_models(all_models, args, client, data_for_db):
    models_to_compare = [m for m in all_models if m != 'BM25']
    use_bm25 = 'BM25' in all_models
    model_instances = {}

    if models_to_compare:
        print("🔄 Инициализация моделей для dense векторов...")
        progress_bar = tqdm(total=len(models_to_compare), desc="Загрузка моделей", unit="модель")

        for model_name in models_to_compare.copy():
            try:
                logger.info(f"Инициализация модели: {model_name}")
                model_instances[model_name] = SentenceTransformer(model_name)
                progress_bar.update(1)
            except Exception as e:
                logger.error(f"Ошибка при инициализации модели {model_name}: {e}")
                models_to_compare.remove(model_name)
        progress_bar.close()
        print(f"✅ Модели инициализированы: {', '.join(models_to_compare)}")

    search_algorithms = {
        "Exact Search": SearchParams(exact=True),
        f"HNSW Users ef={args.hnsw_ef}": SearchParams(hnsw_ef=args.hnsw_ef),
        "HNSW High Precision ef=512": SearchParams(hnsw_ef=512)
    }

    bm25_model = None
    if use_bm25:
        print("🔄 Инициализация модели BM25...")
        logger.info("Инициализация модели BM25")
        bm25_model = 'BM25'

        bm25_collection_name = f"{args.collection_name}_bm25"
        upload_bm25_data(client, bm25_collection_name, data_for_db)
        search_algorithms = {"Exact Search": SearchParams(exact=True)}

    return models_to_compare, bm25_model, model_instances, search_algorithms


def evaluate_dense_models(client, models_to_compare, search_algorithms, args, data_df):
    """
    Выполняет бенчмарк уже загруженных dense-моделей.
    """
    speed_results = {}
    accuracy_results = {}

    for model_name in models_to_compare:
        collection_name = f"{args.collection_name}_{model_name.replace('-', '_')}"
        speed_results[model_name] = {}
        accuracy_results[model_name] = {}

        for algo_name, search_params in search_algorithms.items():
            logger.info(f"🔍 Оценка алгоритма {algo_name} с моделью {model_name}")
            print(f"\n🔍 Оценка алгоритма {algo_name} с моделью {model_name}")

            if algo_name.startswith("HNSW"):
                client.update_collection(
                    collection_name=collection_name,
                    hnsw_config=HnswConfigDiff(
                        m=args.hnsw_m,
                        ef_construct=args.ef_construct,
                    )
                )

            benchmark_results = benchmark_performance(
                client=client,
                collection_name=collection_name,
                test_data=data_df,
                model_name=model_name,
                search_params=search_params,
                top_k_values=[1, 3]
            )

            speed_results[model_name][algo_name] = benchmark_results["speed"]
            accuracy_results[model_name][algo_name] = benchmark_results["accuracy"]

    return speed_results, accuracy_results


def run_dense_benchmark(client, all_models, args, data_for_db, data_df):
    models_to_compare, bm25_model, model_instances, search_algorithms = initialize_models(all_models, args, client,
                                                                                          data_for_db)
    speed_results = {}
    accuracy_results = {}
    upload_dense_model_collections(client, models_to_compare, args, data_for_db)

    # бенчмарк для dense моделей
    if models_to_compare:
        speed_results, accuracy_results = evaluate_dense_models(
            client=client,
            models_to_compare=models_to_compare,
            search_algorithms=search_algorithms,
            args=args,
            data_df=data_df
        )

        if bm25_model != 'BM25':
            print_speed_results(speed_results, models_to_compare)
            print_accuracy_results(accuracy_results, models_to_compare)
            visualize_results(
                speed_results=speed_results,
                accuracy_results=accuracy_results,
                title_prefix="Сравнение производительности RAG системы",
                save_dir="./logs/graphs"
            )

    # бенчмарк для BM25
    bm25_results = None
    if bm25_model:
        bm25_results = run_benchmark_bm25_model(client, args.collection_name, data_for_db, data_df, search_algorithms)
        print_speed_results(speed_results, models_to_compare, bm25_results)
        print_accuracy_results(accuracy_results, models_to_compare, bm25_results)

        visualize_results(
            speed_results=speed_results,
            accuracy_results=accuracy_results,
            bm25_results=bm25_results,
            title_prefix="Сравнение производительности RAG системы",
            save_dir="./logs/graphs"
        )

    logger.info("Бенчмарк завершен успешно")
    print("\n" + "=" * 80)
    print("✅ БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО")
    print("Графики сохранены в директории ./logs/graphs/")
    print("=" * 80)


def main():
    args = parse_args()
    hybrid = args.hybrid
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК БЕНЧМАРКА RAG СИСТЕМЫ")
    print("=" * 80)
    logger.info("Запуск бенчмарка RAG системы")
    # инициализация клиента Qdrant
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    # загрузка данных в соответствии с аргументом limit
    logger.info(f"Загрузка данных с limit={args.limit}")
    print(f"📂 Загрузка данных (limit={args.limit})...")
    data_for_db, data_df = read_data(limit=args.limit)
    logger.info(f"Загружено {len(data_for_db)} документов")
    print(f"✅ Загружено {len(data_for_db)} документов")

    if hybrid == 0:
        # список моделей из аргументов командной строки
        all_models = args.model_names
        logger.info(f"Выбранные модели для сравнения: {', '.join(all_models)}")
        print(f"🔄 Выбранные модели для сравнения: {', '.join(all_models)}")
        run_dense_benchmark(client, all_models, args, data_for_db, data_df)

    # гибридный поиск в гибридной коллекции
    elif hybrid == 1:
        print("\n" + "=" * 80)
        print("🚀 ЗАПУСК БЕНЧМАРКА RAG СИСТЕМЫ С ГИБРИДНЫМ ПОИСКОМ")
        print("=" * 80)
        logger.info("Запуск бенчмарка RAG системы")
        args = parse_args()
        data_for_db, data_df = read_data(limit=args.limit)
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        results_without_rerank, results_with_rerank = run_bench_hybrid(client, data_for_db, data_df)
        print_comparison(results_without_rerank, results_with_rerank)
        visualize_results_rerank(results_without_rerank, results_with_rerank)
        logger.info("Бенчмарк завершен успешно")
        print("\n" + "=" * 80)
        print("✅ БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО")
        print(f"Графики сохранены в директории {GRAPHS_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    main()
