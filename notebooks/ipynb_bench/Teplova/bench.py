import time
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SearchParams, PointStruct


class QdrantAlgorithmBenchmark:

    # Класс для сравнения нескольких поисковых алгоритмов и моделей токенезации

    def __init__(
            self,
            qdrant_url: Optional[str] = None,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "rag01",
            vector_size: int = 384,
            distance: Distance = Distance.COSINE,
            api_key: Optional[str] = None,
    ):

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key)

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.search_algorithms = {}
        self.models = {}

        # создание коллекции

        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )

    # метод для добавления поискового алгоритма
    def add_search_algorithm(
            self,
            name: str,
            search_params: SearchParams,
            description: Optional[str] = None
    ):

        self.search_algorithms[name] = {
            "params": search_params,
            "description": description or ""
        }

    # метод для добавления модели для генерации эмбеддингов
    def add_model(
            self,
            name: str,
            embedding_function: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]],
            description: Optional[str] = None
    ):
        self.models[name] = {
            "function": embedding_function,
            "description": description or ""
        }
    # метод для загрузки коллекции
    def upload_data(
            self,
            data: List[Dict[str, Any]],
            model_name: str,
            text_field: str = "context",
            batch_size: int = 100,
            id_field: str = "id"
    ):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        embedding_function = self.models[model_name]["function"]

        # загрузка батчами
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            texts = [item[text_field] for item in batch]

            try:
                vectors = embedding_function(texts)
                batch_embedding = True
            except:
                vectors = [embedding_function(text) for text in texts]
                batch_embedding = False

            # подготовка вектора
            points = []
            for j, item in enumerate(batch):
                vector = vectors[j] if batch_embedding else vectors[j]

                # используем имеющийся id или создаем его
                if id_field in item and item[id_field] is not None:
                    point_id = item[id_field]
                else:
                    point_id = i + j

                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=item
                ))

            #  загрузка в qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print(f"Uploaded {len(data)} documents with model {model_name}")

# метод формирования результата сравнения
    def run_benchmark(
            self,
            queries: List[str],
            model_names: Optional[List[str]] = None,
            algorithm_names: Optional[List[str]] = None,
            top_k: int = 10,
            ground_truth: Optional[List[List[int]]] = None,
            runs_per_query: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        results = {}

        # указание модели для теста
        models_to_test = list(self.models.keys()) if model_names is None else model_names

        # указание алгоритма
        algos_to_test = list(self.search_algorithms.keys()) if algorithm_names is None else algorithm_names

        for model_name in models_to_test:
            if model_name not in self.models:
                print(f"Модель {model_name} не найдена")
                continue

            model_results = {}
            embedding_function = self.models[model_name]["function"]

            for algo_name in algos_to_test:
                if algo_name not in self.search_algorithms:
                    print(f"Алгоритм {algo_name} не известен")
                    continue

                search_params = self.search_algorithms[algo_name]["params"]

                query_times = []
                precision_scores = []
                recall_scores = []

                for i, query in enumerate(queries):
                    query_times_for_this_query = []

                    # генерация эмбеддинга
                    query_vector = embedding_function(query)

                    # запуск нескольких попыток
                    for _ in range(runs_per_query):
                        # Measure search time
                        start_time = time.time()
                        search_result = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector,
                            search_params=search_params,
                            limit=top_k
                        )
                        end_time = time.time()
                        query_time = end_time - start_time
                        query_times_for_this_query.append(query_time)

                    # вычисляем среднее время
                    query_times.append(np.median(query_times_for_this_query))

                    # расчет точности
                    if ground_truth is not None and i < len(ground_truth):
                        retrieved_ids = [hit.id for hit in search_result]
                        relevant_ids = set(ground_truth)

                        #расчет precision
                        relevant_retrieved = sum(1 for id in retrieved_ids if id in relevant_ids)
                        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
                        precision_scores.append(precision)

                        # расчет recall
                        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 1.0
                        recall_scores.append(recall)

                # записываем результат
                model_results[algo_name] = {
                    "avg_query_time": np.mean(query_times),
                    "median_query_time": np.median(query_times),
                    "max_query_time": np.max(query_times),
                    "min_query_time": np.min(query_times),
                    "total_queries": len(queries),
                }

                if ground_truth is not None:
                    if precision_scores:
                        model_results[algo_name]["avg_precision"] = np.mean(precision_scores)
                    if recall_scores:
                        model_results[algo_name]["avg_recall"] = np.mean(recall_scores)

                    # вычисление F1
                    if precision_scores and recall_scores:
                        f1_scores = []
                        for p, r in zip(precision_scores, recall_scores):
                            if p + r > 0:
                                f1 = 2 * p * r / (p + r)
                            else:
                                f1 = 0
                            f1_scores.append(f1)
                        model_results[algo_name]["avg_f1"] = np.mean(f1_scores)

            results[model_name] = model_results

        return results

    # метод для отрисовки графиков
    def visualize_results(
            self,
            results: Dict[str, Dict[str, Any]],
            metric: str = "avg_query_time",
            title: Optional[str] = None,
            figure_size: Tuple[int, int] = (12, 6),
            sort_by: Optional[str] = None
    ):

        plt.figure(figsize=figure_size)

        models = list(results.keys())
        first_model = next(iter(results.values()))
        algos = list(first_model.keys())

        # сортировка
        if sort_by == 'name':
            algos.sort()
        elif sort_by == metric and metric in first_model[algos[0]]:
            algos.sort(key=lambda algo: first_model[algo].get(metric, 0))

        bar_width = 0.8 / len(models)
        index = np.arange(len(algos))

        for i, model in enumerate(models):
            model_results = results[model]
            values = [model_results[algo].get(metric, 0) for algo in algos]

            # перевод в миллисекунды
            if metric.endswith("_time"):
                values = [v * 1000 for v in values]
                y_label = metric.replace('_', ' ').title() + " (ms)"
            else:
                y_label = metric.replace('_', ' ').title()

            plt.bar(
                index + i * bar_width,
                values,
                bar_width,
                label=model
            )

        plt.xlabel('Алгоритмы поиска')
        plt.ylabel(y_label)
        plt.title(title or f'Сравнение {metric.replace("_", " ").title()} по моделям и алгоритмам')
        plt.xticks(index + bar_width * (len(models) - 1) / 2, algos, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    # метод для очищения удаления коллекции
    def clear_collection(self):

        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
        )
        print(f"Collection {self.collection_name} has been cleared")