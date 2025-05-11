import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from logger_init import setup_paths, setup_logging

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)


def plot_bars_with_labels(index, values, bar_width, color, label, offset, hatch=None, fmt="{:.1f}", y_offset=0.001):
    bars = plt.bar(
        index + offset,
        values,
        bar_width,
        label=label,
        color=color,
        edgecolor='black',
        linewidth=0.5,
        hatch=hatch
    )
    for j, v in enumerate(values):
        if v > 0:
            plt.text(
                index[j] + offset,
                v + y_offset,
                fmt.format(v),
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=45
            )


def visualize_results(speed_results, accuracy_results, bm25_results=None, title_prefix="Результаты бенчмарка",
                      save_dir="./logs/graphs"):
    print(f"\n📊 Создание визуализаций результатов...")
    logger.info("Создание визуализаций результатов")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    has_dense_models = bool(speed_results) and bool(accuracy_results)
    has_bm25 = bm25_results is not None

    if not has_dense_models and not has_bm25:
        logger.warning("Нет результатов для визуализации")
        print("⚠️ Нет результатов для визуализации")
        return

    bm25_has_algos = has_bm25 and isinstance(bm25_results["speed"], dict) and any(
        isinstance(v, dict) for v in bm25_results["speed"].values()
    )

    # --------- Визуализация скорости ---------
    plt.figure(figsize=(12, 7))
    if has_dense_models:
        models = list(speed_results.keys())
        algorithms = list(speed_results[models[0]].keys())
    else:
        models = []
        algorithms = list(bm25_results["speed"].keys()) if bm25_has_algos else ["BM25"]

    n_groups = len(algorithms)
    n_models = len(models) + (1 if has_bm25 else 0)
    bar_width = 0.8 / n_models if n_models else 0.8
    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        values = [speed_results[model][algo]["avg_time"] * 1000 for algo in algorithms]
        plot_bars_with_labels(index, values, bar_width, colors[i], model, i * bar_width)

    if has_bm25:
        i = len(models)
        if bm25_has_algos:
            values = [bm25_results["speed"][algo]["avg_time"] * 1000 for algo in algorithms]
        else:
            value = bm25_results["speed"]["avg_time"] * 1000
            values = [value] * len(algorithms)
        plot_bars_with_labels(index, values, bar_width, colors[i], "BM25", i * bar_width, hatch='//')

    plt.xlabel('Алгоритмы поиска')
    plt.ylabel('Среднее время поиска (мс)')
    plt.title(f"{title_prefix}: Скорость поиска")
    plt.xticks(index + bar_width * (n_models - 1) / 2, algorithms)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    speed_path = f"{save_dir}/speed_comparison_{timestr}.png"
    plt.savefig(speed_path, dpi=300, bbox_inches='tight')
    logger.info(f"График скорости сохранен в {speed_path}")

    # --------- Визуализация точности ---------
    plt.figure(figsize=(12, 7))
    all_top_k = set()

    if has_dense_models:
        models = list(accuracy_results.keys())
        algorithms = list(accuracy_results[models[0]].keys())
        for model in models:
            for algo in algorithms:
                all_top_k.update(accuracy_results[model][algo].keys())
    else:
        models = []
        algorithms = list(bm25_results["accuracy"].keys()) if bm25_has_algos else []

    if has_bm25:
        if bm25_has_algos:
            for algo in bm25_results["accuracy"]:
                all_top_k.update(bm25_results["accuracy"][algo].keys())
        else:
            all_top_k.update(bm25_results["accuracy"].keys())

    top_k_values = sorted(all_top_k)
    if not top_k_values:
        logger.warning("Нет данных для визуализации точности")
        print("⚠️ Нет данных для визуализации точности")
        return

    n_groups = len(top_k_values)
    n_bars = len(models) * len(algorithms) if has_dense_models else 0
    n_bars += len(bm25_results["accuracy"]) if has_bm25 and bm25_has_algos else (1 if has_bm25 else 0)

    bar_width = 0.8 / n_bars
    index = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    i = 0
    if has_dense_models:
        for model in models:
            for algo in algorithms:
                values = [
                    accuracy_results[model][algo].get(k, {}).get("accuracy", 0)
                    for k in top_k_values
                ]
                plot_bars_with_labels(index, values, bar_width, colors[i], f"{model} - {algo}", i * bar_width,
                                      fmt="{:.5f}", y_offset=0.0001)
                i += 1

    if has_bm25:
        if bm25_has_algos:
            for algo in bm25_results["accuracy"]:
                values = [
                    bm25_results["accuracy"][algo].get(k, {}).get("accuracy", 0)
                    for k in top_k_values
                ]
                plot_bars_with_labels(index, values, bar_width, colors[i], f"BM25 - {algo}", i * bar_width, hatch='//',
                                      fmt="{:.5f}", y_offset=0.0001)
                i += 1
        else:
            values = [
                bm25_results["accuracy"].get(k, {}).get("accuracy", 0)
                for k in top_k_values
            ]
            plot_bars_with_labels(index, values, bar_width, colors[i], "BM25", i * bar_width, hatch='//', fmt="{:.3f}",
                                  y_offset=0.01)

    plt.xlabel('Top-K')
    plt.ylabel('Точность')
    plt.title(f"{title_prefix}: Точность поиска")
    plt.xticks(index + bar_width * (n_bars - 1) / 2, top_k_values)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    acc_path = f"{save_dir}/accuracy_comparison_{timestr}.png"
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    logger.info(f"График точности сохранен в {acc_path}")

    print(f"✅ Визуализации сохранены в директории {save_dir}")


def visualize_results_rerank(
        results_without_rerank,
        results_with_rerank,
        top_k_values,
        title_prefix="Сравнение для гибридного поиска с реранкингом и без",
        save_dir=f"{GRAPHS_DIR}"
):
    print(f"\n📊 Создание визуализаций результатов реранкинга...")
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    colors = plt.cm.tab10(np.linspace(0, 1, 2))
    labels = ["Без реранкинга", "С реранкингом"]

    # --- Визуализация времени ---
    plt.figure(figsize=(10, 8))
    speeds = [
        results_without_rerank['speed']['avg_time'] * 1000,
        results_with_rerank['speed']['avg_time'] * 1000
    ]
    index = np.arange(len(speeds))
    bar_width = 0.4

    for i, (label, speed) in enumerate(zip(labels, speeds)):
        plot_bars_with_labels(
            index=np.array([i]),
            values=[speed],
            bar_width=bar_width,
            color=colors[i],
            label=label,
            offset=0,
            fmt="{:.3f}",
            y_offset=0.0001
        )

    plt.xticks(index, labels)
    plt.ylabel("Время (мс)")
    plt.title(f"{title_prefix}: Время поиска")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/speed_comparison_{timestr}_hybrid.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Визуализация точности ---
    plt.figure(figsize=(10, 8))
    acc_before = [results_without_rerank["accuracy"]["before_rerank"][k]["accuracy"] for k in top_k_values]
    acc_after = [results_with_rerank["accuracy"]["after_rerank"][k]["accuracy"] for k in top_k_values]

    index = np.arange(len(top_k_values))
    bar_width = 0.35

    plot_bars_with_labels(
        index=index,
        values=acc_before,
        bar_width=bar_width,
        color=colors[0],
        label=labels[0],
        offset=-bar_width / 2,
        fmt="{:.4f}",
        y_offset=0.0001
    )

    plot_bars_with_labels(
        index=index,
        values=acc_after,
        bar_width=bar_width,
        color=colors[1],
        label=labels[1],
        offset=bar_width / 2,
        fmt="{:.4f}",
        y_offset=0.0001
    )

    plt.xticks(index, [f"Top-{k}" for k in top_k_values])
    plt.ylabel("Точность (Accuracy)")
    plt.title(f"{title_prefix}: Точность поиска")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_comparison_{timestr}_hybrid.png", dpi=300, bbox_inches='tight')
    plt.close()
