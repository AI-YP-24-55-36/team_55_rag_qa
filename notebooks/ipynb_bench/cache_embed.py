import numpy as np
import logging
from pathlib import Path
from load_config import load_config

config = load_config()

BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
EMBEDDINGS_DIR = BASE_DIR / config["paths"]["embeddings_dir"]

# логирование
logger = logging.getLogger('embed')
logger.setLevel(logging.INFO)
logger.propagate = False
file_handler = logging.FileHandler(f'{LOGS_DIR}/embed.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def generate_and_save_embeddings(texts, model, array_name, save_dir=None):

    save_dir = Path(save_dir) if save_dir else EMBEDDINGS_DIR
    save_dir.mkdir(exist_ok=True, parents=True)
    # полный путь к файлу
    file_path = save_dir / f"{array_name}.npy"

    # проверка на существование файла
    if file_path.exists():
        logger.info(f"Файл {file_path} уже существует. Загружаем существующие эмбеддинги.")
        return np.load(file_path)

    # генерация новых эмбеддингов
    logger.info(f"Генерируем новые эмбеддинги для {array_name}...")
    vectors = model.encode(texts, show_progress_bar=False)
    np.save(file_path, vectors)
    logger.info(f"Эмбеддинги успешно сохранены в {file_path}")

    return vectors