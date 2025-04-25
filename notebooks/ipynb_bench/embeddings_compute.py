import numpy as np
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from beir.retrieval.models import SentenceBERT
import logging
from pathlib import Path
import pickle
from load_config import load_config
from read_data_from_csv import read_data

config = load_config()

BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
EMBEDDINGS_DIR = BASE_DIR / config["paths"]["embeddings_dir"]


logger = logging.getLogger('embed')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(f'{LOGS_DIR}/embed.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_embedding_models():
    bm25_model = SparseTextEmbedding("Qdrant/bm25")
    dense_model = SentenceBERT("msmarco-distilbert-base-tas-b")
    colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    return bm25_model, dense_model, colbert_model

bm25_model, dense_model, colbert_model = load_embedding_models()


def build_embeddings(item, bm25_model, dense_model, colbert_model):
    text = item["context"]
    # BM25 sparse vector
    sparse_vector = list(bm25_model.query_embed(text))
    sparse_embedding = sparse_vector[0] if sparse_vector else None
    # Dense vector
    # dense_embedding = dense_model.encode(text).tolist()
    dense_embedding = dense_model.encode_corpus([{"text": text}], convert_to_tensor=False)[0]
    # ColBERT vector
    colbert_embedding = list(colbert_model.embed(text))[0]

    return sparse_embedding, dense_embedding, colbert_embedding

def generate_emb():
    sparce, dense, colbert = [], [], []
    data_for_db, data_df = read_data(limit=-1)

    for item in tqdm(data_for_db):
        sparse_embedding, dense_embedding, colbert_embedding = build_embeddings(item, bm25_model, dense_model, colbert_model)
        sparce.append(sparse_embedding)
        dense.append(dense_embedding)
        colbert.append(colbert_embedding)

    return sparce, dense, colbert

def pad_to_fixed_length(matrix, target_len=1024):
    current_len = matrix.shape[0]
    dim = matrix.shape[1]

    if current_len > target_len:
        # обрезаем, если слишком длинный
        return matrix[:target_len]
    elif current_len < target_len:
        # добавим нули
        pad = np.zeros((target_len - current_len, dim), dtype=matrix.dtype)
        return np.vstack((matrix, pad))
    else:
        return matrix

def memmap_emb(
    dense_output_path=f'{EMBEDDINGS_DIR}/dense_embeddings.memmap',
    colbert_output_path=f'{EMBEDDINGS_DIR}/colbert_embeddings.memmap',
    sparse_output_path=f'{EMBEDDINGS_DIR}/sparse_embeddings.pkl',
    colbert_tokens=1024,  # фиксированное количество токенов
    dtype='float32'
):
    # Получаем эмбеддинги
    sparse, dense, colbert = generate_emb()

    num_texts = len(dense)
    dense_dim = len(dense[0])
    colbert_shape = colbert[0].shape  # (tokens, 128)

    tokens, colbert_dim = colbert_shape

    print(f"💾 Сохраняем dense эмбеддинги в {dense_output_path}...")
    dense_memmap = np.memmap(dense_output_path, dtype=dtype, mode='w+', shape=(num_texts, dense_dim))
    for idx, vector in enumerate(dense):
        dense_memmap[idx] = vector
    del dense_memmap  # flush

    print(f"💾 Сохраняем colbert эмбеддинги в {colbert_output_path}")
    colbert_memmap = np.memmap(colbert_output_path, dtype=dtype, mode='w+',
                               shape=(num_texts, colbert_tokens, colbert_dim))
    for idx, matrix in enumerate(colbert):
        padded_matrix = pad_to_fixed_length(matrix, target_len=colbert_tokens)
        colbert_memmap[idx] = padded_matrix
    del colbert_memmap

    print(f"💾 Сохраняем sparse эмбеддинги (BM25) в {sparse_output_path}...")
    with open(sparse_output_path, 'wb') as f:
        pickle.dump(sparse, f)

    print("✅ Все эмбеддинги успешно сохранены!")


if __name__ == '__main__':
    memmap_emb()