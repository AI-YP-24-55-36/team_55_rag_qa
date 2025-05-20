from fastembed import TextEmbedding

EMBEDDING_MODELS = {
    "jina-embeddings-v2-base-en": TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en"),  # 768
    "snowflake-arctic-embed-s": TextEmbedding(model_name="snowflake/snowflake-arctic-embed-s"),  # 384
    "mxbai-embed-large-v1": TextEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1"),      # 1024
    "multilingual-e5-large": TextEmbedding(model_name="intfloat/multilingual-e5-large"),         # 1024  The model intfloat/multilingual-e5-large now uses mean pooling instead of CLS embedding.
}

MODEL_VECTOR_SIZES = {
        "jina-embeddings-v2-base-en": 768,
        "snowflake-arctic-embed-s": 384,
        "snowflake-arctic-embed-m": 768,
        "mxbai-embed-large-v1": 1024,
        "multilingual-e5-large" : 1024,
}