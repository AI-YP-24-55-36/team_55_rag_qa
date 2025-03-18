import pandas as pd
import logging
import numpy as np
from pathlib import Path

# Настройка логгера для текущего модуля
logger = logging.getLogger('read_data')
logger.setLevel(logging.INFO)
logger.propagate = False  # Отключаем передачу логов родительским логгерам

# Создание обработчика для записи логов в файл
Path('./logs').mkdir(exist_ok=True)
file_handler = logging.FileHandler('./logs/read_data.log')
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)


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


