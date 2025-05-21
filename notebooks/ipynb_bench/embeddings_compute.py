import numpy as np
from tqdm import tqdm
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from models_init import EMBEDDING_MODELS

import pickle
import torch
from read_data_from_csv import read_data
from logger_init import setup_paths, setup_logging

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)


def load_embedding_models():
    bm25_model = SparseTextEmbedding("Qdrant/bm25")
    colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    dense_models = EMBEDDING_MODELS

    return bm25_model, colbert_model, dense_models

#
def build_embeddings(item, bm25_model, colbert_model, dense_models):
    text = item["context"]

    # BM25 sparse vector
    sparse_vector = list(bm25_model.query_embed(text))
    sparse_embedding = sparse_vector[0] if sparse_vector else None

    # Dense vectors (все модели)
    dense_embeddings = {}
    for model_name, model in dense_models.items():
        emb = list(model.embed(text, normalize=True))[0]
        dense_embeddings[model_name] = emb

    # ColBERT vector
    colbert_embedding = list(colbert_model.embed(text))[0]
    return sparse_embedding, dense_embeddings, colbert_embedding

def generate_emb():
    sparse_list = []
    colbert_list = []
    dense_dict = {name: [] for name in dense_models}

    data_for_db, data_df = read_data(limit=11000)

    for item in tqdm(data_for_db):
        sparse_embedding, dense_embeddings, colbert_embedding = build_embeddings(
            item, bm25_model, colbert_model, dense_models
        )

        sparse_list.append(sparse_embedding)
        colbert_list.append(colbert_embedding)

        for name, emb in dense_embeddings.items():
            dense_dict[name].append(emb)

    return sparse_list, dense_dict, colbert_list


def pad_to_fixed_length(matrix, target_len=512):
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
        output_dir=EMBEDDINGS_DIR,
        colbert_tokens=256,
        dtype='float32'
):
    sparse, dense_dict, colbert = generate_emb()

    num_texts = len(colbert)
    colbert_shape = colbert[0].shape
    tokens, colbert_dim = colbert_shape

    # Сохраняем dense эмбеддинги всех моделей
    for model_name, dense_vectors in dense_dict.items():
        dense_dim = len(dense_vectors[0])
        dense_output_path = f"{output_dir}/{model_name}.memmap"
        print(f"💾 Сохраняем dense эмбеддинги ({model_name}) в {dense_output_path}...")

        dense_memmap = np.memmap(dense_output_path, dtype=dtype, mode='w+', shape=(num_texts, dense_dim))
        for idx, vector in enumerate(dense_vectors):
            dense_memmap[idx] = vector
        del dense_memmap

    # ColBERT
    colbert_output_path = f"{output_dir}/colbert_embeddings.memmap"
    print(f"💾 Сохраняем ColBERT эмбеддинги в {colbert_output_path}...")
    colbert_memmap = np.memmap(colbert_output_path, dtype=dtype, mode='w+',
                               shape=(num_texts, colbert_tokens, colbert_dim))
    for idx, matrix in enumerate(colbert):
        padded_matrix = pad_to_fixed_length(matrix, target_len=colbert_tokens)
        colbert_memmap[idx] = padded_matrix
    del colbert_memmap

    # Sparse
    sparse_output_path = f"{output_dir}/sparse_embeddings.pkl"
    print(f"💾 Сохраняем sparse эмбеддинги (BM25) в {sparse_output_path}...")
    with open(sparse_output_path, 'wb') as f:
        pickle.dump(sparse, f)

    print("✅ Все эмбеддинги успешно сохранены!")


if __name__ == '__main__':
    bm25_model, colbert_model, dense_models = load_embedding_models()
    memmap_emb()
