import pandas as pd
import numpy as np
from pathlib import Path
from logger_init import setup_paths, setup_logging

BASE_DIR, LOGS_DIR, GRAPHS_DIR, OUTPUT_DIR, EMBEDDINGS_DIR = setup_paths()
logger = setup_logging(LOGS_DIR, OUTPUT_DIR)


class DatasetError(Exception):
    """Исключение для ошибок при работе с датасетом"""
    pass


def read_data(file_path='full_dataset.csv', limit=100, random_seed=42):
    """Чтение и подготовка данных из CSV файла"""
    if not Path(file_path).exists():
        raise DatasetError(f"Файл датасета не найден: {file_path}")

    try:
        logger.info(f"Чтение данных из {file_path}...")
        rag_dataset = pd.read_csv(file_path)

        if 'context' not in rag_dataset.columns or 'question' not in rag_dataset.columns:
            raise DatasetError(
                "Датасет должен содержать колонки 'context' и 'question'")

        # Предобработка текстов
        df = rag_dataset.copy()

        # Удаление строк с пустыми значениями
        initial_size = len(df)
        df = df.dropna(subset=['context', 'question'])
        if len(df) == 0:
            raise DatasetError("После удаления пустых значений датасет пуст")

        logger.info(
            f"Удалено строк с пустыми значениями: {initial_size - len(df)}")

        # Приведение к нижнему регистру
        df['context'] = df['context'].str.lower()
        df['question'] = df['question'].str.lower()
        if 'answer' in df.columns:
            df['answer'] = df['answer'].str.lower()

        # Очистка текста
        df['context'] = df['context'].str.replace('\n', ' ').str.strip()
        df['question'] = df['question'].str.strip()

        # Удаление дубликатов
        before_dedup = len(df)
        df.drop_duplicates(subset=['question'], keep='first', inplace=True)
        logger.info(f"Удалено дубликатов: {before_dedup - len(df)}")

        # Сброс индексов
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Итоговое количество записей: {len(df)}")

        # Ограничиваем размер датасета
        if limit > 0 and limit < len(df):
            np.random.seed(random_seed)
            indices = np.random.choice(len(df), size=limit, replace=False)
            df = df.iloc[indices].reset_index(drop=True)
            logger.info(f"Датасет ограничен до {limit} записей")

        # Создаем данные для загрузки в базу
        data_for_db = [
            {
                "id": i,
                "context": context,
                "question": question
            }
            for i, (context, question) in enumerate(zip(df.context, df.question))
        ]

        logger.info(f"Подготовлено {len(data_for_db)} записей")

        return data_for_db, df

    except pd.errors.EmptyDataError:
        raise DatasetError(f"Файл {file_path} пуст")
    except pd.errors.ParserError:
        raise DatasetError(
            f"Ошибка при парсинге файла {file_path}. Убедитесь, что это корректный CSV файл")
    except Exception as e:
        raise DatasetError(f"Ошибка при обработке датасета: {str(e)}")
