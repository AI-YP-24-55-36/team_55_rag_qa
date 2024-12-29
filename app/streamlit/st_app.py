import streamlit as st
from validate_df import validate_csv
from eda import plot_length, length, plot_top_words, plot_wordcloud, prep, plot_tsne
from project_logger import log_and_display

st.set_page_config(
    page_title="RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


def main():
    # загрузка заголовка приложения
    st.title("Обучение модели для чат-бота на основе RAG")
    st.markdown("""
    ### Возможности приложения
    - загрузка датасета и анализ данных
    - препроцессинг данных
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
            log_and_display("Файл успешно загружен и проверен!", level="success", display_func=st.success)
            data = valid_df
            st.write("Превью - первые 5 строк:")
            # data = data[:5000]
            st.dataframe(data.head(5))

        if valid_df is not None:
            if st.sidebar.checkbox("Показать информацию о данных"):
                st.success(f"Всего {len(data)} строк")
                st.success(f"Всего {len(data.columns)} столбца")

            if st.sidebar.checkbox("Проверить есть ли дубликаты"):
                columns = data.columns
                if data.duplicated().values.any():
                    st.warning("В датасете есть полные дубликаты")
                elif data.duplicated(subset=columns[0]).values.any():
                    st.warning(f"В датасете есть дубликаты в колонке {columns[0]}")
                elif data.duplicated(subset=columns[1]).values.any():
                    st.warning(f"В датасете есть дубликаты в колонке {columns[1]}")
                elif data.duplicated(subset=columns[2]).values.any():
                    st.warning(f"В датасете есть дубликаты в колонке в колонке {columns[2]}")
                else:
                    st.success("В датасете нет дубликатов")

            if st.sidebar.checkbox("Проверить есть ли пропуски в данных"):
                # Проверка наличия пропущенных значений
                if data.isnull().values.any():
                    st.warning("В датасете есть пропущенные значения.")
                    # Получение списка строк с пропущенными значениями
                    missing_rows = data[data.isnull().any(axis=1)].index.tolist()
                    miss = data[data.isnull().any(axis=1)]
                    # Вывод списка строк с пропущенными значениями
                    st.success("Список строк с пропущенными значениями:")
                    for row in missing_rows:
                        st.write(f"Строка {row}")
                    if st.sidebar.button("строки с пропусками"):
                        if missing_rows:
                            st.warning("Строки с пропущенными значениями:")
                            st.write(miss.head(10))
                        else:
                            st.write("В датасете нет пропущенных значений.")
                else:
                    st.success("В датасете нет пропущенных значений.")

            st.sidebar.title("Графики")

            def clear_other_checkboxes(checked_key):
                for key in st.session_state.keys():
                    if key != checked_key and st.session_state[key]:
                        st.session_state[key] = False

            if uploaded_file is not None:


                if st.sidebar.checkbox("Длины текстов", key="graph1", on_change=clear_other_checkboxes,
                                       args=("graph1",)):
                    new_data = length(data)
                    col_len = new_data.columns[-3:].to_list()
                    fig = plot_length(new_data, col_len)
                    st.plotly_chart(fig, use_container_width=True)
                    log_and_display("график с длинами слов отрисован успешно", level="info")
                    # st.pyplot(fig)
            if uploaded_file is not None:
                if st.sidebar.checkbox("Частотность слов", key="graph2", on_change=clear_other_checkboxes,
                                       args=("graph2",)):
                    cols = data.columns
                    fig, ax = plot_top_words(data[cols[0]])
                    log_and_display("график с частотностью слов отрисован успешно", level="info")
                    st.pyplot(fig)

            if uploaded_file is not None:
                if st.sidebar.checkbox("Облако слов", key="graph3", on_change=clear_other_checkboxes, args=("graph3",)):
                    cols = data.columns
                    fig, ax = plot_wordcloud(data[cols[0]])
                    log_and_display("облако слов отрисовано успешно", level="info")
                    st.pyplot(fig)

            if uploaded_file is not None:

                if st.sidebar.checkbox("t-SNE для топ-200 слов", key="graph4", on_change=clear_other_checkboxes,
                                       args=("graph4",),
                                       help="Если корпус слов большой, то вычисление потребует некоторого времени"):
                    cols = data.columns
                    fig, ax = plot_tsne(data[cols[0]])
                    log_and_display("t-SNE отрисован успешно", level="info")
                    st.pyplot(fig)

            st.sidebar.title("Препроцессинг")

            if uploaded_file is not None:
                if st.sidebar.button("Удалить дубликаты"):
                    columns = data.columns
                    if data.duplicated().values.any():
                        data.drop_duplicates(keep='first', inplace=True)
                        log_and_display(f"Строки с дубликатами удалены. Стало всего {len(data)} строк", level="success",
                                        display_func=st.success)
                    else:
                        st.warning("В датасете нет дубликатов")

            if uploaded_file is not None:
                if st.sidebar.button("Удалить пропуски"):
                    if data.isnull().values.any():
                        data = data.dropna()
                        log_and_display(f"Строки с пропусками удалены. Стало всего {len(data)} строк", level="success",
                                        display_func=st.success)
                    else:
                        st.warning("В датасете нет пропущенных значений")

            if uploaded_file is not None:
                if st.sidebar.button("Очистить текст"):
                    cols = data.columns
                    data["clear_text"] = data[cols[0]].apply(lambda x: prep(x))
                    log_and_display(
                        f"Текст очищен от стоп-слов, от символов не являющихся буквами и цифрами, приведен к нижнему регистру, эту колонку можно подавать в модель для обучения",
                        level="success",
                        display_func=st.success)
                    st.dataframe(data["clear_text"].head(10))

            st.sidebar.title("Модель")
            if st.sidebar.checkbox("Настроить", key="model", on_change=clear_other_checkboxes, args=("model",)):
                st.sidebar.subheader("Параметры векторизации")

                def on_slider_change_max_df():
                    st.write(f"Выбранное значение max_df: {st.session_state.slider_value_max_df}")
                    value = st.session_state.slider_value_max_df
                    return value

                st.sidebar.slider(
                    "max_df",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    key="slider_value_max_df",
                    on_change=on_slider_change_max_df,
                )

                def on_slider_change_min_df():
                    st.write(f"Выбранное значение min_df: {st.session_state.slider_value_min_df}")
                    value = st.session_state.slider_value_min_df
                    return value

                st.sidebar.slider(
                    "min_df",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key="slider_value_min_df",
                    on_change=on_slider_change_min_df,
                )

                def on_slider_change_max_features():
                    st.write(f"Выбранное значение max_features: {st.session_state.max_features}")
                    value = st.session_state.max_features
                    return value

                st.sidebar.slider(
                    "max_features",
                    min_value=1000,
                    max_value=30000,
                    value=9000,
                    step=500,
                    key="max_features",
                    on_change=on_slider_change_max_features,
                )

                def on_sublinear_tf():
                    st.write(f"Выбранное значение sublinear_tf: {st.session_state.sublinear_tf}")
                    value = st.session_state.sublinear_tf
                    return value

                st.sidebar.radio(
                    "sublinear_tf",
                    [True, False],
                    index=0,
                    key="sublinear_tf",
                    on_change=on_sublinear_tf,
                )

                def on_ngram():
                    st.write(f"Выбранное значение ngram_range: {st.session_state.ngram_range}")
                    value = st.session_state.ngram_range
                    return value

                st.sidebar.radio(
                    "ngram_range",
                    [(1, 1), (1, 2), (1, 3)],
                    index=0,
                    key="ngram_range",
                    on_change=on_ngram,
                )

                hyperparameters = {}
                hyperparameters["max_df"] = on_slider_change_max_df()
                hyperparameters["min_df"] = on_slider_change_min_df()
                hyperparameters["max_features"] = on_slider_change_max_features()
                hyperparameters["smooth_idf"] = on_sublinear_tf()
                hyperparameters["sublinear_tf"] = on_ngram()
                log_and_display(f"гиперпараметры модели {hyperparameters}", level="info")

                st.sidebar.subheader("Выбор метрики близости")

                def on_distance():
                    st.write(f"Метрика: {st.session_state.distance}")
                    value = st.session_state.distance
                    return value

                st.sidebar.radio(
                    "distance",
                    ["models.Distance.COSINE", "models.Distance.EUCLID"],
                    index=0,
                    key="distance",
                    on_change=on_distance,
                )

                qdrant_parameters = {}
                distance = on_distance()
                qdrant_parameters["distance"] = distance
                log_and_display(f"гиперпараметры qdrant {qdrant_parameters}", level="info")

                st.sidebar.subheader("Обучение модели")

                if st.sidebar.button("Параметры"):
                    st.success(f"Модель")
                else:
                    st.warning("Модель не определена")

                if st.sidebar.button("Обучить"):
                    st.success(f"Модель обучена")
                else:
                    st.warning("Не получилось")

                if st.sidebar.button("Точность"):
                    st.success(f"Точность %")
                else:
                    st.warning("Нет модели")

            def text_form(key):
                if "textarea" not in st.session_state:
                    st.session_state.textarea = ""
                st.text_area("Введите текст для проверки работы модели",
                             height=200,
                             value="",
                             placeholder="Введите или скопируйте текст",
                             key=key)

            if st.sidebar.checkbox("Инференс", key="infer", on_change=clear_other_checkboxes, args=("infer",)):
                text_form("textarea")

                if st.button('Отправить текст'):
                    if len(st.session_state.textarea):
                        st.success('Текст отправлен в модель')
                    else:
                        st.warning('Поле пустое')
                if st.session_state.textarea:
                    st.write(st.session_state.textarea)
                    test = st.session_state.textarea
                    # inference  = model(test)
                    log_and_display("Предикт выполнен успешно", level="success")

            if st.sidebar.checkbox("Повторить", key="repeat", on_change=clear_other_checkboxes, args=("repeat",)):
                text_form("repeating")
                if st.button('Отправить текст'):
                    if len(st.session_state.repeating):
                        st.success('Текст отправлен в модель')
                    else:
                        st.warning('Поле пустое')
                if st.session_state.repeating:
                    st.write(st.session_state.repeating)
                    test = st.session_state.repeating
                    # inference  = model(test)
                    log_and_display("Предикт выполнен успешно", level="success")


if __name__ == "__main__":
    main()
