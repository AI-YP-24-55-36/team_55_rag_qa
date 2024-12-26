import streamlit as st
import pandas as pd
from validate_df import validate_csv
import io
import toml
from eda import plot_length, length, plot_top_words, plot_wordcloud
import logging
from logging.handlers import RotatingFileHandler
import sys
import os



st.set_page_config(
    page_title="RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# primary_color = theme_settings.get("primary_color", "#8FAFBE")
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         font-family: sans-serif;
#         background-color: {primary_color};
#         color: white;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

file_handler = RotatingFileHandler(
    'logs/app.log',           # Имя файла лога
    maxBytes=1024*1024,  # Максимальный размер файла
    backupCount=5         # Количество backup-файлов
)

# Настройка формата логирования
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)


def main():

    # загрузка заголовка приложения
    st.title("Обучение модели для чат-бота на основе RAG")
    st.markdown("""
    ### Возможности приложения
    - загрузка датасета и анализ данных
    - конфигурирование и обучение модели
    - получение инференса
    """)

    st.sidebar.title("Информация")

    st.header("Загрузка данных")
    st.markdown("""
       **Формат датасета** - 3 колонки с текстами  
       `контекст(context)`, `вопрос(question)`, `ответ(answer)`
        """)
    st.markdown("ссылка на [пример датасета](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)")
    uploaded_file = st.file_uploader("Выберите CSV-файл, максимальный размер файла 200 Мб", type="csv")

    if uploaded_file is not None:
        # Валидация файла
        valid_df = validate_csv(uploaded_file)
        # Если валидация прошла успешно
        if valid_df is not None:
            st.success("Файл успешно загружен и проверен!")
            data = valid_df
            logging.info("файл датасета загружен")
            st.write("Превью - первые 5 строк:")
            st.dataframe(data.head(5))

        if valid_df is not None:
            if st.sidebar.checkbox("Показать информацию о данных"):
                st.write(f"Всего {len(data)} строк")
                st.write(f"Всего {len(data.columns)} столбца")


            if st.sidebar.checkbox("Проверить есть ли дубликаты"):
                columns = data.columns
                if data.duplicated().values.any():
                    st.write("В датасете есть полные дубликаты")
                elif data.duplicated(subset=columns[0]).values.any():
                    st.write(f"В датасете есть дубликаты в колонке {columns[0]}")
                elif data.duplicated(subset=columns[1]).values.any():
                    st.write(f"В датасете есть дубликаты в колонке {columns[1]}")
                elif data.duplicated(subset=columns[2]).values.any():
                    st.write(f"В датасете есть дубликаты в колонке в колонке {columns[2]}")
                else:
                    st.write("В датасете нет дубликатов")

            if st.sidebar.checkbox("Проверить есть ли пропуски в данных"):
                # Проверка наличия пропущенных значений
                if data.isnull().values.any():
                    st.write("В датасете есть пропущенные значения.")

                    # Получение списка строк с пропущенными значениями
                    missing_rows = data[data.isnull().any(axis=1)].index.tolist()
                    miss = data[data.isnull().any(axis=1)]
                    # Вывод списка строк с пропущенными значениями
                    st.write("Список строк с пропущенными значениями:")
                    for row in missing_rows:
                        st.write(f"Строка {row}")
                    if st.sidebar.button("строки с пропусками"):
                        if missing_rows:
                            st.write("Строки с пропущенными значениями:")
                            st.write(miss)
                        else:
                            st.write("В датасете нет пропущенных значений.")
                else:
                    st.write("В датасете нет пропущенных значений.")

            st.sidebar.title("Графики")

            if uploaded_file is not None:
                if st.sidebar.checkbox("Длины текстов"):
                    new_data = length(data)
                    col_len = new_data.columns[-3:].to_list()
                    fig, ax = plot_length(new_data, col_len)
                    logging.info("график с длинами слов отрисован успешно")
                    st.pyplot(fig)
            if uploaded_file is not None:
                if st.sidebar.checkbox("Частотность слов"):
                    cols = data.columns
                    fig, ax = plot_top_words(data[cols[0]])
                    logging.info("график с частотностью слов отрисован успешно")
                    st.pyplot(fig)
                    # plot_top_words(df['context'], 'context')

            if uploaded_file is not None:
                if st.sidebar.checkbox("Облако слов"):
                    cols = data.columns
                    fig, ax = plot_wordcloud(data[cols[0]])
                    logging.info("облако слов отрисовано успешно")
                    st.pyplot(fig)

            st.sidebar.title("Препроцессинг")

            st.sidebar.title("Обучение")

            st.sidebar.title("Инференс")



if __name__ == "__main__":
    main()