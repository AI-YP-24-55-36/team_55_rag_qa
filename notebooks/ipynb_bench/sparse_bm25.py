import datetime
import logging
import sys
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import models
from log_output import Tee
from load_config import load_config

config = load_config()
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
OUTPUT_DIR = BASE_DIR / config["paths"]["output_dir"]
EMBEDDINGS_DIR = BASE_DIR / config["paths"]["embeddings_dir"]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Tee(f"{OUTPUT_DIR}/log_{timestamp}.txt")

logger = logging.getLogger('hybrid')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(f'{LOGS_DIR}/hybrid.log')
file_handler.setLevel(logging.INFO)

# формат логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def upload_bm25_data(client, collection_name, data):
    """Загрузка данных в Qdrant с использованием встроенного BM25"""

    logger.info(f"Загрузка {len(data)} документов в коллекцию {collection_name} с использованием BM25")
    # проверка, существует ли коллекция
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        client.delete_collection(collection_name)
        logger.info(f"Коллекция {collection_name} удалена")

    # создание коллекцию с BM25-индексом
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=0,) # отключение построение графа
    ),

    logger.info(f"Коллекция {collection_name} создана с поддержкой BM25")

    # инициализация модели
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    points = []
    # создание поинтов
    for item in data:
        vector = list(bm25_embedding_model.query_embed(item["context"]))
        if vector:
            sparse_embedding = vector[0]
            points.append(
                models.PointStruct(
                    id=item["id"],
                    payload= item,
                    vector={
                        "bm25": {
                            "values": sparse_embedding.values.tolist(),
                            "indices": sparse_embedding.indices.tolist()
                        }
                    }
                )
            )

    # загрузка поинтов
    client.upload_points(
        collection_name=collection_name,
        points=points
    )

    logger.info(f"Загрузка данных завершена для коллекции {collection_name}")