import streamlit as st
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler(
    'logs/app.log',           # Имя файла лога
    maxBytes=10*1024*1024,  # Максимальный размер файла
    backupCount=5         # Количество backup-файлов
)

# Настройка формата логирования
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def validate_csv(uploaded_file):
    try:
        max_file_size = 200 * 1024 * 1024  # 200 МБ
        if uploaded_file.size == 0 or uploaded_file.size > max_file_size:
            raise ValueError("Файл датасета не подходит по размеру")
            logging.error("Файл не проходит по размеру")

        else:
            # Чтение файла
            df = pd.read_csv(uploaded_file)
            # Проверка на пустое содержимое
            if df.empty:
                st.error("Файл не содержит данных")
                logging.error("Файл пустой")
                return None
            # Проверка количества столбцов
            if len(df.columns) != 3:
                st.error(f"Требуется ровно 3 столбца. Текущее количество: {len(df.columns)}")
                logging.error("Датасет содержит неправильное кол-во столбцов")
                return None

            # Проверка типов столбцов
            non_text_columns = []
            for col in df.columns:
                # Проверка на тип данны в столбце
                if df[col].dtype == 'object':
                    logging.info("колонки подходящего формата")
                else: non_text_columns.append(col)
            # Если есть столбцы, которые не удалось преобразовать
            if non_text_columns:
                st.error(f"Следующие столбцы не являются текстовыми: {non_text_columns}")
                logging.error("Датасет содержит данные не того формата")
                return None

            return df

    except pd.errors.EmptyDataError:
        st.error("Файл пуст или не может быть прочитан")
        logging.error("Файл пуст или не может быть прочитан")

        return None

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
        logging.error("Ошибка при обработке файла")

        return None

    except ValueError as ve:
        st.error(str(ve))
        logging.error(f"Ошибка загрузки файла: {ve}")

        return None

    except AttributeError:
        st.error("Некорректный файл")
        logging.error("Некорректный файл")
        return None
