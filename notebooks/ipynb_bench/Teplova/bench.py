import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import datetime
from tqdm import tqdm
from qdrant_client import models
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

def benchmark_tfidf(client, collection_name, test_data, model, search_params=None, top_k_values=[1, 3]):
    print(
        f"\n🔍 Запуск оценки производительности TF-IDF для коллекции '{collection_name}'")
    logger.info(
        f"Запуск оценки производительности TF-IDF для коллекции '{collection_name}'")

    # Результаты
    results = {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
    }

    # Получаем максимальное значение top_k для поиска
    max_top_k = max(top_k_values)

    # Общее количество запросов
    total_queries = len(test_data)
    logger.info(
        f"Оценка производительности TF-IDF для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска TF-IDF...")

    # Создаем прогресс-бар
    progress_bar = tqdm(total=total_queries,
                        desc="Обработка запросов TF-IDF", unit="запрос")

    # Преобразуем все запросы в векторы заранее
    query_texts = test_data['question'].tolist()
    query_vectors = model.transform(query_texts)

    # Обрабатываем каждый запрос отдельно
    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        # Получаем вектор запроса
        query_vector = query_vectors[idx]
        query_indices = query_vector.indices.tolist()
        query_values = query_vector.data.tolist()

        # Измеряем время поиска
        start_time = time.time()

        # Выполняем поиск
        search_results = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=query_indices,
                values=query_values,
            ),
            using="text",
            limit=max_top_k,
            search_params=search_params
        )

        end_time = time.time()
        query_time = end_time - start_time
        results["speed"]["query_times"].append(query_time)

        # Оцениваем точность для разных значений top_k
        found_contexts = [point.payload.get(
            'context', '') for point in search_results.points]

        # Проверяем точность для каждого значения top_k
        for k in top_k_values:
            # Увеличиваем общее количество запросов
            results["accuracy"][k]["total"] += 1

            # Проверяем, найден ли правильный контекст в первых k результатах
            if true_context in found_contexts[:k]:
                results["accuracy"][k]["correct"] += 1
                logger.info(
                    f"TF-IDF Запрос {idx}: '{query_text[:50]}...' - Контекст найден в top-{k} ✓")
            else:
                logger.info(
                    f"TF-IDF Запрос {idx}: '{query_text[:50]}...' - Контекст не найден в top-{k} ✗")

        # Обновляем прогресс-бар
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()

    # Вычисляем статистику скорости
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(
            query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)

    # Удаляем промежуточные данные о времени запросов
    del results["speed"]["query_times"]

    # Вычисляем точность для каждого значения top_k
    for k in top_k_values:
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        accuracy = correct / total if total > 0 else 0
        results["accuracy"][k]["accuracy"] = accuracy

        logger.info(
            f"TF-IDF Точность поиска (top-{k}): {accuracy:.4f} ({correct}/{total})")

    # Выводим статистику скорости
    logger.info(
        f"TF-IDF Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(
        f"TF-IDF Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(
        f"TF-IDF Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(
        f"TF-IDF Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")

    print(
        f"✅ Оценка производительности TF-IDF завершена для коллекции '{collection_name}'")

    return results


def benchmark_performance(client, collection_name, test_data, model, search_params=None, top_k_values=[1, 3]):
    print(
        f"\n🔍 Запуск оценки производительности для коллекции '{collection_name}'")
    logger.info(
        f"Запуск оценки производительности для коллекции '{collection_name}'")

    # Результаты
    results = {
        "speed": {
            "avg_time": 0,
            "median_time": 0,
            "max_time": 0,
            "min_time": 0,
            "query_times": []
        },
        "accuracy": {k: {"correct": 0, "total": 0, "accuracy": 0} for k in top_k_values}
    }

    # Получаем максимальное значение top_k для поиска
    max_top_k = max(top_k_values)

    # Общее количество запросов
    total_queries = len(test_data)
    logger.info(f"Оценка производительности для {total_queries} запросов")
    print(f"⏱️  Измерение скорости и точности поиска...")

    # Создаем прогресс-бар
    progress_bar = tqdm(total=total_queries,
                        desc="Обработка запросов", unit="запрос")

    # Обрабатываем каждый запрос отдельно
    for idx, row in test_data.iterrows():
        query_text = row['question']
        true_context = row['context']

        # Генерируем эмбеддинг для запроса
        query_vector = model.encode(query_text, show_progress_bar=False)

        # Измеряем время поиска
        start_time = time.time()

        # Выполняем поиск
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            using="context",
            search_params=search_params,
            limit=max_top_k  # Используем максимальное значение top_k
        )

        end_time = time.time()
        query_time = end_time - start_time
        results["speed"]["query_times"].append(query_time)

        # Оцениваем точность для разных значений top_k
        found_contexts = [point.payload.get(
            'context', '') for point in search_results.points]

        # Проверяем точность для каждого значения top_k
        for k in top_k_values:
            # Увеличиваем общее количество запросов
            results["accuracy"][k]["total"] += 1

            # Проверяем, найден ли правильный контекст в первых k результатах
            if true_context in found_contexts[:k]:
                results["accuracy"][k]["correct"] += 1
                logger.info(
                    f"Запрос {idx}: '{query_text[:50]}...' - Контекст найден в top-{k} ✓")
            else:
                logger.info(
                    f"Запрос {idx}: '{query_text[:50]}...' - Контекст не найден в top-{k} ✗")

        # Обновляем прогресс-бар
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()

    # Вычисляем статистику скорости
    query_times = results["speed"]["query_times"]
    if query_times:
        results["speed"]["avg_time"] = sum(query_times) / len(query_times)
        results["speed"]["median_time"] = sorted(
            query_times)[len(query_times) // 2]
        results["speed"]["max_time"] = max(query_times)
        results["speed"]["min_time"] = min(query_times)

    # Удаляем промежуточные данные о времени запросов
    del results["speed"]["query_times"]

    # Вычисляем точность для каждого значения top_k
    for k in top_k_values:
        correct = results["accuracy"][k]["correct"]
        total = results["accuracy"][k]["total"]
        accuracy = correct / total if total > 0 else 0
        results["accuracy"][k]["accuracy"] = accuracy

        logger.info(
            f"Точность поиска (top-{k}): {accuracy:.4f} ({correct}/{total})")

    # Выводим статистику скорости
    logger.info(
        f"Среднее время поиска: {results['speed']['avg_time'] * 1000:.2f} мс")
    logger.info(
        f"Медианное время поиска: {results['speed']['median_time'] * 1000:.2f} мс")
    logger.info(
        f"Максимальное время поиска: {results['speed']['max_time'] * 1000:.2f} мс")
    logger.info(
        f"Минимальное время поиска: {results['speed']['min_time'] * 1000:.2f} мс")

    print(
        f"✅ Оценка производительности завершена для коллекции '{collection_name}'")

    return results


def visualize_results(speed_results, accuracy_results, tfidf_results=None, title_prefix="Результаты бенчмарка", save_dir=f"{GRAPHS_DIR}/graphs"):
    print(f"\n📊 Создание визуализаций результатов...")
    logger.info("Создание визуализаций результатов")

    # Создаем директорию для сохранения графиков, если она не существует
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Проверяем, есть ли у нас результаты для обычных моделей
    has_dense_models = bool(speed_results) and bool(accuracy_results)
    # Проверяем, есть ли у нас результаты для TF-IDF
    has_tfidf = tfidf_results is not None

    # Если нет ни одного результата, выходим
    if not has_dense_models and not has_tfidf:
        logger.warning("Нет результатов для визуализации")
        print("⚠️ Нет результатов для визуализации")
        return

    # Определяем формат данных TF-IDF
    tfidf_has_algos = False
    if has_tfidf:
        # Проверяем, есть ли у TF-IDF разные алгоритмы
        if isinstance(tfidf_results["speed"], dict) and any(isinstance(v, dict) for v in tfidf_results["speed"].values()):
            tfidf_has_algos = True

    # 1. Визуализация скорости поиска
    plt.figure(figsize=(12, 7))

    # Если есть результаты для обычных моделей
    if has_dense_models:
        models = list(speed_results.keys())
        algorithms = list(speed_results[models[0]].keys())
    else:
        # Если есть только TF-IDF, создаем пустые списки для моделей
        models = []
        # Для TF-IDF берем алгоритмы из его результатов, если они есть
        if tfidf_has_algos:
            algorithms = list(tfidf_results["speed"].keys())
        else:
            # Иначе создаем фиктивный алгоритм
            algorithms = ["TF-IDF"]

    # Определяем количество групп и ширину столбцов
    n_groups = len(algorithms)

    # Определяем количество моделей для отображения
    n_models = len(models)
    if has_tfidf and tfidf_has_algos:
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

    # Добавляем TF-IDF, если он есть
    if has_tfidf:
        i = len(models)  # Индекс для TF-IDF

        if tfidf_has_algos:
            # Если у TF-IDF есть разные алгоритмы, отображаем их как отдельные столбцы
            values = [tfidf_results["speed"][algo]
                      ["avg_time"] * 1000 for algo in algorithms]

            plt.bar(
                index + i * bar_width,
                values,
                bar_width,
                label="TF-IDF",
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
            tfidf_value = tfidf_results["speed"]["avg_time"] * 1000

            # Повторяем значение для каждого алгоритма
            tfidf_values = [tfidf_value] * len(algorithms)

            plt.bar(
                index + i * bar_width,
                tfidf_values,
                bar_width,
                label="TF-IDF",
                color=colors[i],
                edgecolor='black',
                linewidth=0.5,
                hatch='//'  # Добавляем штриховку для выделения
            )

            # Добавляем значения над столбцами
            for j, v in enumerate(tfidf_values):
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
        # Если есть только TF-IDF, создаем пустые списки для моделей
        models = []
        # Для TF-IDF берем алгоритмы из его результатов, если они есть
        if tfidf_has_algos:
            algorithms = list(tfidf_results["accuracy"].keys())
        else:
            algorithms = []

    # Если есть TF-IDF, добавляем его top_k значения
    if has_tfidf:
        if tfidf_has_algos:
            # Если у TF-IDF есть разные алгоритмы
            for algo in tfidf_results["accuracy"].keys():
                all_top_k.update(tfidf_results["accuracy"][algo].keys())
        else:
            # Старый формат
            all_top_k.update(tfidf_results["accuracy"].keys())

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

    if has_tfidf:
        if tfidf_has_algos:
            n_bars += len(tfidf_results["accuracy"].keys())
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

    # Добавляем TF-IDF, если он есть
    if has_tfidf:
        if tfidf_has_algos:
            # Если у TF-IDF есть разные алгоритмы
            for algo in tfidf_results["accuracy"].keys():
                values = []
                for k in top_k_values:
                    if k in tfidf_results["accuracy"][algo]:
                        values.append(
                            tfidf_results["accuracy"][algo][k]["accuracy"])
                    else:
                        values.append(0)  # Если нет данных для этого top_k

                plt.bar(
                    index + i * bar_width,
                    values,
                    bar_width,
                    label=f"TF-IDF - {algo}",
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
                if k in tfidf_results["accuracy"]:
                    values.append(tfidf_results["accuracy"][k]["accuracy"])
                else:
                    values.append(0)  # Если нет данных для этого top_k

            plt.bar(
                index + i * bar_width,
                values,
                bar_width,
                label="TF-IDF",
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
