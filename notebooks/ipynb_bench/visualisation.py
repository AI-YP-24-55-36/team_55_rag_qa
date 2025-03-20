import datetime
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def visualize_results(speed_results, accuracy_results, bm25_results=None, title_prefix="Результаты бенчмарка",
                      save_dir=f"{GRAPHS_DIR}"):
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
        # Для BM25 берем алгоритмы из его результатов, если они есть
        if bm25_has_algos:
            algorithms = list(bm25_results["accuracy"].keys())
        else:
            algorithms = []

    # Если есть BM25, добавляем его top_k значения
    if has_bm25:
        if bm25_has_algos:
            # Если у BM25 есть разные алгоритмы
            for algo in bm25_results["accuracy"].keys():
                all_top_k.update(bm25_results["accuracy"][algo].keys())
        else:
            # Старый формат
            all_top_k.update(bm25_results["accuracy"].keys())

    top_k_values = sorted(list(all_top_k))

    # Если нет значений top_k, выходим
    if not top_k_values:
        logger.warning("Нет данных для визуализации точности")
        print("⚠️ Нет данных для визуализации точности")
        return

    # Определяем количество групп и ширину столбцов
    n_groups = len(top_k_values)

    # Определяем количество столбцов
    if has_dense_models:
        n_bars = len(models) * len(algorithms)
    else:
        n_bars = 0

    if has_bm25:
        if bm25_has_algos:
            n_bars += len(bm25_results["accuracy"].keys())
        else:
            n_bars += 1

    # Если нет столбцов, выходим
    if n_bars == 0:
        logger.warning("Нет данных для визуализации точности")
        print("⚠️ Нет данных для визуализации точности")
        return

    bar_width = 0.8 / n_bars

    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    i = 0
    # Отображаем столбцы для обычных моделей, если они есть
    if has_dense_models:
        for model in models:
            for algo in algorithms:
                values = []
                for k in top_k_values:
                    if k in accuracy_results[model][algo]:
                        values.append(
                            accuracy_results[model][algo][k]["accuracy"])
                    else:
                        values.append(0)  # Если нет данных для этого top_k

                plt.bar(
                    index + i * bar_width,
                    values,
                    bar_width,
                    label=f"{model} - {algo}",
                    color=colors[i],
                    edgecolor='black',
                    linewidth=0.5
                )

                # Добавляем значения над столбцами
                for j, v in enumerate(values):
                    if v > 0:  # Показываем только ненулевые значения
                        plt.text(
                            index[j] + i * bar_width,
                            v + 0.01,
                            f"{v:.3f}",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45
                        )

                i += 1

    # Добавляем BM25, если он есть
    if has_bm25:
        if bm25_has_algos:
            # Если у BM25 есть разные алгоритмы
            for algo in bm25_results["accuracy"].keys():
                values = []
                for k in top_k_values:
                    if k in bm25_results["accuracy"][algo]:
                        values.append(
                            bm25_results["accuracy"][algo][k]["accuracy"])
                    else:
                        values.append(0)  # Если нет данных для этого top_k

                plt.bar(
                    index + i * bar_width,
                    values,
                    bar_width,
                    label=f"BM25 - {algo}",
                    color=colors[i],
                    edgecolor='black',
                    linewidth=0.5,
                    hatch='//'  # Добавляем штриховку для выделения
                )

                # Добавляем значения над столбцами
                for j, v in enumerate(values):
                    if v > 0:  # Показываем только ненулевые значения
                        plt.text(
                            index[j] + i * bar_width,
                            v + 0.01,
                            f"{v:.3f}",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45
                        )

                i += 1
        else:
            # Старый формат
            values = []
            for k in top_k_values:
                if k in bm25_results["accuracy"]:
                    values.append(bm25_results["accuracy"][k]["accuracy"])
                else:
                    values.append(0)  # Если нет данных для этого top_k

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
                if v > 0:  # Показываем только ненулевые значения
                    plt.text(
                        index[j] + i * bar_width,
                        v + 0.01,
                        f"{v:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=45
                    )

    plt.xlabel('Top-K')
    plt.ylabel('Точность')
    plt.title(f"{title_prefix}: Точность поиска")
    plt.xticks(index + bar_width * (n_bars - 1) / 2, top_k_values)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    accuracy_save_path = f"{save_dir}/accuracy_comparison_{timestr}.png"
    plt.savefig(accuracy_save_path, dpi=300, bbox_inches='tight')
    logger.info(f"График точности сохранен в {accuracy_save_path}")

    print(f"✅ Визуализации сохранены в директории {save_dir}")
