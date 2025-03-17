import numpy as np
import logging
from pathlib import Path
import os

logger = logging.getLogger('embed')
logger.setLevel(logging.INFO)
logger.propagate = False

Path('./logs').mkdir(exist_ok=True)
file_handler = logging.FileHandler('./logs/embed.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)

def generate_and_save_embeddings(texts, model, array_name, save_dir="embeddings"):

    # Создаем директорию, если она не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Полный путь к файлу
    file_path = os.path.join(save_dir, f"{array_name}.npy")

    # Проверка на существование файла
    if os.path.exists(file_path):
        logger.info(f"Файл {file_path} уже существует. Загружаем существующие эмбеддинги.")

        return np.load(file_path)
    # Генерация новых эмбеддингов
    logger.info(f"Генерируем новые эмбеддинги для {array_name}...")
    vectors = model.encode(texts, show_progress_bar=False)

    # Сохранение массива
    np.save(file_path, vectors)
    logger.info(f"Эмбеддинги успешно сохранены в {file_path}")

    return vectors