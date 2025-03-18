
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import datetime
from log_output import Tee
from load_config import load_config

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
OUTPUT_DIR = BASE_DIR / config["paths"]["output_dir"]


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Tee(f"{OUTPUT_DIR}/log_{timestamp}.txt")

logger = logging.getLogger('bench')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(f'{LOGS_DIR}/bench.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)


def visualize_results_bm25(speed_results, accuracy_results, bm25_results=None, title_prefix="Результаты бенчмарка",
                      save_dir=f"{GRAPHS_DIR}/graphs"):
    print(f"\n📊 Создание визуализаций результатов...")
    logger.info("Создание визуализаций результатов")

    # Создаем директорию для сохранения графиков, если она не существует
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Проверяем, есть ли у нас результаты для обычных моделей
    has_dense_models = bool(speed_results) and bool(accuracy_results)
    # Проверяем, есть ли у нас результаты для BM25
    has_bm25 = bm25_results is not None

    # Если нет ни одного результата, выходим
    if not has_dense_models and not has_bm25:
        logger.warning("Нет результатов для визуализации")
        print("⚠️ Нет результатов для визуализации")
        return

    # Определяем формат данных BM25
    bm25_has_algos = False
    if has_bm25:
        # Проверяем, есть ли у BM25 разные алгоритмы
        if isinstance(bm25_results["speed"], dict) and any(isinstance(v, dict) for v in bm25_results["speed"].values()):
            bm25_has_algos = True

    # 1. Визуализация скорости поиска
    plt.figure(figsize=(12, 7))

    # Если есть результаты для обычных моделей
    if has_dense_models:
        models = list(speed_results.keys())
        algorithms = list(speed_results[models[0]].keys())
    else:
        # Если есть только BM25, создаем пустые списки для моделей
        models = []
        # Для BM25 берем алгоритмы из его результатов, если они есть
        if bm25_has_algos:
            algorithms = list(bm25_results["speed"].keys())
        else:
            # Иначе создаем фиктивный алгоритм
            algorithms = ["BM25"]

    # Определяем количество групп и ширину столбцов
    n_groups = len(algorithms)

    # Определяем количество моделей для отображения
    n_models = len(models)
    if has_bm25 and bm25_has_algos:
        n_models += 1

    bar_width = 0.8 / n_models if n_models > 0 else 0.8

    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    # Отображаем столбцы для обычных моделей, если они есть
    if has_dense_models:
        for i, (model, color) in enumerate(zip(models, colors)):
            values = [speed_results[model][algo]["avg_time"] *
                      1000 for algo in algorithms]  # в миллисекундах

            plt.bar(
                index + i * bar_width,
                values,
                bar_width,
                label=model,
                color=color,
                edgecolor='black',
                linewidth=0.5
            )

            # Добавляем значения над столбцами
            for j, v in enumerate(values):
                plt.text(
                    index[j] + i * bar_width,
                    v + 0.1,
                    f"{v:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )

    # Добавляем BM25, если он есть
    if has_bm25:
        i = len(models)  # Индекс для BM25

        if bm25_has_algos:
            # Если у BM25 есть разные алгоритмы, отображаем их как отдельные столбцы
            values = [bm25_results["speed"][algo]["avg_time"] * 1000 for algo in algorithms]

            plt.bar(
                index + i * bar_width,
                values,
                bar_width,
                label="BM25",
                color=colors[i],
                edgecolor='black',
                linewidth=0.5,
                hatch='//'  # Добавляем штриховку для выделения
            )

            # Добавляем значения над столбцами
            for j, v in enumerate(values):
                plt.text(
                    index[j] + i * bar_width,
                    v + 0.1,
                    f"{v:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )
        else:
            # Старый формат - одно значение для всех алгоритмов
            bm25_value = bm25_results["speed"]["avg_time"] * 1000

            # Повторяем значение для каждого алгоритма
            bm25_values = [bm25_value] * len(algorithms)

            plt.bar(
                index + i * bar_width,
                bm25_values,
                bar_width,
                label="BM25",
                color=colors[i],
                edgecolor='black',
                linewidth=0.5,
                hatch='//'  # Добавляем штриховку для выделения
            )

            # Добавляем значения над столбцами
            for j, v in enumerate(bm25_values):
                plt.text(
                    index[j] + i * bar_width,
                    v + 0.1,
                    f"{v:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )

    plt.xlabel('Алгоритмы поиска')
    plt.ylabel('Среднее время поиска (мс)')
    plt.title(f"{title_prefix}: Скорость поиска")
    plt.xticks(index + bar_width * (n_models - 1) / 2, algorithms)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    speed_save_path = f"{save_dir}/speed_comparison_{timestr}.png"
    plt.savefig(speed_save_path, dpi=300, bbox_inches='tight')
    logger.info(f"График скорости сохранен в {speed_save_path}")

    plt.figure(figsize=(12, 7))

    # Определяем все возможные значения top_k
    all_top_k = set()

    # Если есть обычные модели, собираем их top_k значения
    if has_dense_models:
        models = list(accuracy_results.keys())
        algorithms = list(accuracy_results[models[0]].keys())

        for model in models:
            for algo in algorithms:
                all_top_k.update(accuracy_results[model][algo].keys())
    else:
        # Если есть только BM25, создаем пустые списки для моделей
        models = []
        # Для BM25 берем алгоритмы из его результатов,