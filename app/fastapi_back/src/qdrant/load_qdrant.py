from typing import List
import time
from qdrant_client import QdrantClient, models
from scipy.sparse import csr_matrix
from ..logger import qdrant_logger


def save_vectors_batch(client, source_texts: List[str], vectors: csr_matrix, collection_name: str = "default"):
    """Функция сохранения векторов"""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )
    qdrant_logger.info("collection %s created", collection_name)
    points = []
    for i in range(vectors.shape[0]):
        indices = vectors[i].indices.tolist()
        data = vectors[i].data.tolist()
        points.append(
            models.PointStruct(
                id=i,
                payload={
                    'source_text': source_texts[i]
                },
                vector={
                    'text': models.SparseVector(
                        indices=indices, values=data
                    )
                },
            )
        )
    client.upload_points(
        collection_name=collection_name,
        points=points,
        parallel=4,
        max_retries=3,
    )
    qdrant_logger.info("collection %s filled", collection_name)


def search_similar_texts(
    client: QdrantClient,
    query_vec,
    collection_name: str = "default",  # добавить дефолтную коллекцию
    limit: int = 1
) -> List[dict]:
    """Функция поиска схожих текстов"""
    query_indices = query_vec[0].indices.tolist()
    query_data = query_vec[0].data.tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=query_indices,
            values=query_data,
        ),
        using="text",
        limit=limit
    )

    found_texts = [
        {
            "source_text": point.payload.get("source_text"),
            "score": point.score,
            "point_id": point.id
        }
        for point in results.points
    ]
    qdrant_logger.info("texts found")
    return found_texts


# оптимизировать
def check_questions(client, df, model, collection_name) -> dict[str, float | List[float]]:
    """Функция поиска вопросов"""
    correct = 0
    query_text = df['question']
    query_vecs = model.transform(query_text)
    timings = []
    for idx, row in df.iterrows():
        query_vec = query_vecs[idx]
        query_indices = query_vec.indices.tolist()
        query_data = query_vec.data.tolist()
        start_time = time.time()
        result = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=query_indices,
                values=query_data,
            ),
            using="text",
            limit=1,
            search_params=models.SearchParams(hnsw_ef=128, exact=False)
        )
        end_time = time.time()
        timings.append(end_time - start_time)
        top_n = len(result.points)
        res = [result.points[i].payload['source_text'] for i in range(top_n)]
        if row['context'] in res:
            correct += 1
    accuracy = correct / len(df)
    return {'accuracy': accuracy, 'timings': timings}
