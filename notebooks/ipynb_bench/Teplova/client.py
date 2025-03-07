from qdrant_client.http.models import Distance, SearchParams, HnswConfigDiff
from sentence_transformers import SentenceTransformer
from bench import QdrantAlgorithmBenchmark
from read_data_from_csv import read_data

# инициализация класса
benchmark = QdrantAlgorithmBenchmark(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="rag",
    vector_size=384,
    distance=Distance.COSINE
)

# данные для теста
test_data, test_queries, ground_truth = read_data()

# инициализация класса
benchmark = QdrantAlgorithmBenchmark(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="rag",
    vector_size=384,
    distance=Distance.COSINE
)

# данные для теста
test_data, test_queries, ground_truth = read_data()

# основная функция с запуском бенчмарка
def main(model_st, model_name_enc):

    # определяем алгоритмы поиска
    # точный поиск (exact search)
    benchmark.add_search_algorithm(
        name="Exact Search",
        search_params=SearchParams(
            exact=True
        ),
        description="Точный поиск по векторам"
    )

    # HNSW с разными параметрами
    benchmark.add_search_algorithm(
        name="HNSW Default",
        search_params=SearchParams(
            hnsw_ef=128
        ),
        description="HNSW поиск с ef=128"
    )

    benchmark.add_search_algorithm(
        name="HNSW High Precision",
        search_params=SearchParams(
            hnsw_ef=512
        ),
        description="HNSW поиск с ef=512 для более высокой точности"
    )

    model_st = model_st


    def get_sentence_transformer_embedding(text):
        if isinstance(text, list):
            return model_st.encode(text).tolist()
        return model_st.encode(text).tolist()

    benchmark.add_model(
        name=model_name_enc,
        embedding_function=get_sentence_transformer_embedding,
        description=model_name_enc
    )

    #  очистим коллекцию
    benchmark.clear_collection()

    # загрузка данных с помощью первой модели
    benchmark.upload_data(
        data=test_data,
        model_name=model_name_enc,
        text_field="context",
        batch_size=100
    )

    # запуск бенчмарка
    results = benchmark.run_benchmark(
        queries=test_queries,
        model_names=[model_name_enc],
        algorithm_names=["Exact Search", "HNSW Default", "HNSW High Precision"],
        top_k=10,
        runs_per_query=3
    )

    # Вывод и визуализация результатов
    print("Результаты бенчмарка:")
    for model, model_results in results.items():
        print(f"\nМодель: {model}")
        for algo, metrics in model_results.items():
            print(f"  Алгоритм: {algo}")
            for metric, value in metrics.items():
                if "time" in metric:
                    print(f"    {metric}: {value * 1000:.2f} мс")
                else:
                    print(f"    {metric}: {value}")

    # визуализация результатов
    benchmark.visualize_results(
        results=results,
        metric="avg_query_time",
        title="Сравнение времени поиска для разных алгоритмов",
        sort_by="name"
    )


    # запуск бенчмарка с ground truth для оценки точности
    results_with_metrics = benchmark.run_benchmark(
        queries=test_queries,
        model_names=[model_name_enc],
        algorithm_names=["Exact Search", "HNSW Default", "HNSW High Precision"],
        top_k=10,
        ground_truth=ground_truth,
        runs_per_query=3
    )

    # визуализация метрик точности
    benchmark.visualize_results(
        results=results_with_metrics,
        metric="avg_precision",
        title="Сравнение точности поиска (Precision)",
        sort_by=None
    )

    benchmark.visualize_results(
        results=results_with_metrics,
        metric="avg_recall",
        title="Сравнение полноты поиска (Recall)",
        sort_by=None
    )

    # обновляем параметры HNSW для коллекции
    benchmark.client.update_collection(
        collection_name=benchmark.collection_name,
        hnsw_config=HnswConfigDiff(
            m=16,  # увеличение количества соседей
            ef_construct=200,  # увеличение параметра ef при построении
        )
    )

    #  повторный тест для оценки влияния изменений
    results_after_config_update = benchmark.run_benchmark(
        queries=test_queries,
        model_names=[model_name_enc],
        algorithm_names=["HNSW Default", "HNSW High Precision"],
        top_k=10,
        runs_per_query=3
    )

    # сравнение производительность до и после оптимизации
    print("\nСравнение производительности до и после оптимизации HNSW:")
    for algo in ["HNSW Default", "HNSW High Precision"]:
        before = results[model_name_enc][algo]["avg_query_time"] * 1000
        after = results_after_config_update[model_name_enc][algo]["avg_query_time"] * 1000

        improvement = (before - after) / before * 100
        if before > after:
            print(f"{algo}: {before:.2f} мс -> {after:.2f} мс (ускорение на {improvement:.2f}%)")
        else:
            print(f"{algo}: {before:.2f} мс -> {after:.2f} мс (замедление на {improvement:.2f}%)")


if __name__ == "__main__":
    #  модель
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    model_name_enc = 'all-MiniLM-L6-v2'

    # вызываем main с передачей модели и её имени
    main(model_st, model_name_enc)