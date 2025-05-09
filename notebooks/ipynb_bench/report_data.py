from logger_init import setup_paths, setup_logging

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)

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

def calculate_speed_stats(results):
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)
    del results["speed"]["query_times"]

def compute_final_accuracy(results):
    for k, data in results["accuracy"].items():
        correct = data["correct"]
        total = data["total"]
        data["accuracy"] = correct / total if total > 0 else 0

def log_topk_accuracy(results, top_k_values):
    for k in top_k_values:
        acc = results["accuracy"][k]["accuracy"]
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        logger.info(f"Точность поиска (top-{k}): {acc:.4f} ({correct}/{total})")

def log_speed_stats(results):
    logger.info(f"Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(f"Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(f"Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(f"Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")



def print_speed_results(speed_results, models_to_compare, bm25_results=None):
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ СКОРОСТИ ПОИСКА")
    print("=" * 80)

    if models_to_compare:
        for model_name in models_to_compare:
            print(f"\nМодель: {model_name}")
            for algo_name, result in speed_results[model_name].items():
                print(f" Алгоритм: {algo_name}")
                print(f" Среднее время: {result['avg_time'] * 1000:.2f} мс")
                print(f" Медианное время: {result['median_time'] * 1000:.2f} мс")
                print(f" Максимальное время: {result['max_time'] * 1000:.2f} мс")
                print(f" Минимальное время: {result['min_time'] * 1000:.2f} мс")

    if bm25_results:
        print(f"\nМодель: BM25")
        for algo_name, result in bm25_results["speed"].items():
            print(f"  Алгоритм: {algo_name}")
            print(f"  Среднее время: {result['avg_time'] * 1000:.2f} мс")
            print(f"  Медианное время: {result['median_time'] * 1000:.2f} мс")
            print(f"  Максимальное время: {result['max_time'] * 1000:.2f} мс")
            print(f"  Минимальное время: {result['min_time'] * 1000:.2f} мс")

def print_accuracy_results(accuracy_results, models_to_compare, bm25_results=None):
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ ТОЧНОСТИ ПОИСКА")
    print("=" * 80)

    # Результаты для dense моделей
    if models_to_compare:
        for model_name in models_to_compare:
            print(f"\nМодель: {model_name}")
            for algo_name in accuracy_results[model_name].keys():
                print(f"  Алгоритм: {algo_name}")
                for k in [1, 3]:
                    if k in accuracy_results[model_name][algo_name]:
                        result = accuracy_results[model_name][algo_name][k]
                        print(
                            f"    Top-{k}: Точность = {result['accuracy']:.4f} "
                            f"({result['correct']}/{result['total']})"
                        )

    # Результаты для BM25
    if bm25_results:
        print(f"\nМодель: BM25")
        for algo_name in bm25_results["accuracy"].keys():
            print(f"  Алгоритм: {algo_name}")
            for k in [1, 3]:
                if k in bm25_results["accuracy"][algo_name]:
                    result = bm25_results["accuracy"][algo_name][k]
                    print(
                        f"    Top-{k}: Точность = {result['accuracy']:.4f} "
                        f"({result['correct']}/{result['total']})"
                    )