import streamlit as st
import pandas as pd
from project_logger import log_and_display

def validate_csv(uploaded_file):

    try:
        max_file_size = 200 * 1024 * 1024  # 200 МБ
        if uploaded_file.size == 0 or uploaded_file.size > max_file_size:
            raise (ValueError("Файл датасета не подходит по размеру"))
            log_and_display("Файл датасета не подходит по размеру", level="error", display_func=st.error)

        else:
            # Чтение файла
            df = pd.read_csv(uploaded_file)
            # Проверка на пустое содержимое
            if df.empty:
                (log_and_display("Файл не содержит данных", level="error", display_func=st.error))
                return None
            # Проверка количества столбцов
            if len(df.columns) != 3:
                log_and_display(f"Требуется ровно 3 столбца. Текущее количество: {len(df.columns)}", level="error", display_func=st.error)
                return None

            # Проверка типов столбцов
            non_text_columns = []
            for col in df.columns:
                # Проверка на тип данны в столбце
                if df[col].dtype != 'object':
                    non_text_columns.append(col)

                # Если есть столбцы, которые не удалось преобразовать
                if non_text_columns:
                    log_and_display(f"Следующие столбцы не являются текстовыми: {non_text_columns}", level="error", display_func=st.error)
                    return None
                # else: non_text_columns.append(col)
        return df

    except pd.errors.EmptyDataError:
        log_and_display(f"Файл пуст или не может быть прочитан", level="error", display_func=st.error)
        return None

    except Exception as e:
        log_and_display(f"Ошибка при обработке файла: {e}", level="error", display_func=st.error)
        return None

    except ValueError as ve:
        st.error(str(ve))
        log_and_display(f"Ошибка загрузки файла: {ve}", level="error", display_func=st.error)
        return None

    except AttributeError:
        log_and_display(f"Некорректный файл", level="error", display_func=st.error)
        return None