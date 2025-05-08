
import datetime
import logging
import sys
from pathlib import Path
from log_output import Tee
from load_config import load_config

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