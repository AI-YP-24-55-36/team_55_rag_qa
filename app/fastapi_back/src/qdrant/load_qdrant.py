from qdrant_client import QdrantClient, models
from scipy.sparse import csr_matrix
from typing import List


def save_vectors_batch(client, source_texts: List[str], vectors: csr_matrix, collection_name: str = "default"):
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
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def search_similar_texts(
        client: QdrantClient,
        query_vec,
        collection_name: str = "default", #  добавить дефолтную коллекцию
        limit: int = 1
    ) -> List[dict]:
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
    
    return found_texts