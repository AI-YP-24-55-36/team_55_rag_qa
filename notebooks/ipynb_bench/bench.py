import datetime
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm
from log_output import Tee
from load_config import load_config
from report_data import (init_results, evaluate_top_k_accuracy,
                         compute_speed_stats, compute_final_accuracy,
                         log_topk_accuracy, log_speed_stats)

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
OUTPUT_DIR = BASE_DIR / config["paths"]["output_dir"]


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Tee(f"{OUTPUT_DIR}/log_{timestamp}.txt")
# логгирование
logger = logging.getLogger('bench')
logger.setLevel(logging.INFO)
logger.propagate = False
file_handler = logging.FileHandler(f'{LOGS_DIR}/bench.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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

def benchmark_performance(client, collection_name, test_data, model, search_params=None, top_k_values=[1, 3]):
    print(f"\n🔍 Запуск оценки производительности для коллекции '{collection_name}'")
    logger.info(f"Запуск оценки производительности для коллекции '{collection_name}'")

    results = init_results(top_k_values)
    max_top_k = max(top_k_values)
    total_queries = len(test_data)

    logger.info(f"Оценка производительности для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска...")

    progress_bar = tqdm(total=total_queries, desc="Обработка запросов", unit="запрос")

    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']
        query_vector = model.encode(query_text, show_progress_bar=False)

        search_results, query_time = run_query(client, collection_name, query_vector, search_params, max_top_k)
        results["speed"]["query_times"].append(query_time)

        found_contexts = [point.payload.get('context', '') for point in search_results.points]

        evaluate_top_k_accuracy(results, found_contexts, true_context, top_k_values, query_text, idx)

        progress_bar.update(1)

    progress_bar.close()

    compute_speed_stats(results)
    compute_final_accuracy(results)
    log_topk_accuracy(results, top_k_values)
    log_speed_stats(results)

    print(f"✅ Оценка производительности завершена для коллекции '{collection_name}'")
    return results