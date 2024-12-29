import streamlit as st
from validate_df import validate_csv
from eda import plot_length, length, plot_top_words, plot_wordcloud, prep, plot_tsne, plot_bench
from project_logger import log_and_display
import requests
import time

st.set_page_config(
    page_title="RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/api/v1/models"


def clear_other_checkboxes(checked_key):
    for key in st.session_state.keys():
        if key != checked_key and st.session_state[key]:
            st.session_state[key] = False


def text_form(key):
    if "textarea" not in st.session_state:
        st.session_state.textarea = ""
    st.text_area("Введите текст для проверки работы модели",
                 height=200,
                 value="",
                 placeholder="Введите или скопируйте текст",
                 key=key)
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
            # переименование колонок в соотвествии с требованием бэка
            data = data.rename(columns={
                data.columns[0]: 'context',
                data.columns[1]: 'question',
                data.columns[2]: 'answer'
            })
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
            if st.sidebar.checkbox("Длины текстов", key="graph1", on_change=clear_other_checkboxes,
                                   args=("graph1",)):
                new_data = length(data)
                col_len = new_data.columns[-3:].to_list()
                fig = plot_length(new_data, col_len)
                st.plotly_chart(fig, use_container_width=True)
                log_and_display("график с длинами слов отрисован успешно", level="info")
            if st.sidebar.checkbox("Частотность слов", key="graph2", on_change=clear_other_checkboxes,
                                   args=("graph2",)):
                cols = data.columns
                fig, ax = plot_top_words(data[cols[0]])
                log_and_display("график с частотностью слов отрисован успешно", level="info")
                st.pyplot(fig)

            if st.sidebar.checkbox("Облако слов", key="graph3", on_change=clear_other_checkboxes, args=("graph3",)):
                cols = data.columns
                fig, ax = plot_wordcloud(data[cols[0]])
                log_and_display("облако слов отрисовано успешно", level="info")
                st.pyplot(fig)


            if st.sidebar.checkbox("t-SNE для топ-200 слов", key="graph4", on_change=clear_other_checkboxes,
                                   args=("graph4",),
                                   help="Если корпус слов большой, то вычисление потребует некоторого времени"):
                cols = data.columns
                fig, ax = plot_tsne(data[cols[0]])
                log_and_display("t-SNE отрисован успешно", level="info")
                st.pyplot(fig)

            st.sidebar.title("Препроцессинг")
            st.sidebar.markdown("""
                                - удаление дубликатов и пропусков
                                - предобработка текста
                                - отправка данных на сервер
                                """
                                )

            if st.sidebar.button("Очистить и отправить"):
                columns = data.columns
                if data.duplicated().values.any():
                    data.drop_duplicates(keep='first', inplace=True)
                    log_and_display(f"Строки с дубликатами удалены. Стало всего {len(data)} строк", level="success",
                                    display_func=st.success)
                else:
                    st.warning("В датасете нет дубликатов")

                if data.isnull().values.any():
                    data = data.dropna()
                    log_and_display(f"Строки с пропусками удалены. Стало всего {len(data)} строк", level="success",
                                    display_func=st.success)
                else:
                    st.warning("В датасете нет пропущенных значений")
                data["context"] = data[columns[0]].apply(lambda x: prep(x))
                data["question"] = data[columns[1]].apply(lambda x: prep(x))
                data["answer"] = data[columns[2]].apply(lambda x: prep(x))

                log_and_display(
                    f"Текст очищен от стоп-слов, от символов не являющихся буквами и цифрами", level="success",
                    display_func=st.success)
                st.dataframe(data.head(10))
                payload = {}
                payloads = data.to_dict("records")
                data_name = uploaded_file.name
                payload["datasets"] = {data_name: payloads}
                response = requests.post(f"{API_URL}/load_dataset", json=payload)
                mess = response.json()[0]["message"]
                if response.status_code == 201:
                    log_and_display(f"{mess}", level="success", display_func=st.success)
                else:
                    log_and_display(f"Ошибка при запросе API: {response.status_code}", level="error",
                                    display_func=st.error)

            if st.sidebar.title("Модель"):
                if st.sidebar.checkbox("Настроить", key="model", on_change=clear_other_checkboxes, args=("model",)):
                    st.sidebar.subheader("Параметры векторизации")

                    def on_slider_change_max_df():
                        value = st.session_state.slider_value_max_df
                        return value

                    st.sidebar.slider(
                        "max_df",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.85,
                        step=0.1,
                        key="slider_value_max_df",
                        on_change=on_slider_change_max_df,
                    )

                    def on_slider_change_min_df():
                        value = st.session_state.slider_value_min_df
                        return value

                    st.sidebar.slider(
                        "min_df",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="slider_value_min_df",
                        on_change=on_slider_change_min_df,
                    )

                    def on_slider_change_max_features():
                        value = st.session_state.max_features
                        return value

                    st.sidebar.slider(
                        "max_features",
                        min_value=20000,
                        max_value=100000,
                        value=50000,
                        step=1000,
                        key="max_features",
                        on_change=on_slider_change_max_features,
                    )

                    def on_sublinear_tf():
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
                        value = st.session_state.ngram_range
                        return value

                    st.sidebar.radio(
                        "ngram_range",
                        [(1, 1), (1, 2), (1, 3)],
                        index=1,
                        key="ngram_range",
                        on_change=on_ngram,
                    )

                    hyperparameters = {}
                    hyperparameters["ngram_range"] = on_ngram()
                    hyperparameters["max_df"] = on_slider_change_max_df()
                    hyperparameters["min_df"] = on_slider_change_min_df()
                    hyperparameters["max_features"] = on_slider_change_max_features()
                    hyperparameters["sublinear_tf"] = on_sublinear_tf()
                    log_and_display(f"гиперпараметры модели {hyperparameters}", level="info")
                    model_id = st.sidebar.text_input("model_id", max_chars=20)

                    if st.sidebar.button("Сохранить параметры"):
                        fit_p = {}
                        fit_save = []
                        fit_p["hyperparameters"] = hyperparameters
                        fit_p["model_id"] = model_id
                        fit_p["ml_model_type"] = "tf-idf"
                        fit_p["dataset_nm"] = uploaded_file.name
                        st.warning('Параметры отправлены в модель, ждем ответ...')
                        fit_save.append(fit_p)
                        response = requests.post(f"{API_URL}/fit_save", json=fit_save)
                        if isinstance(response.json(), dict):
                            log_and_display(f"{response.json()}", level="warning", display_func=st.warning)
                        else:
                            mess = response.json()[0]["message"]
                            if response.status_code == 201:
                                log_and_display(f"{mess}", level="success", display_func=st.success)
                            else:
                                log_and_display(f"Ошибка при запросе API: {response.status_code}", level="error", display_func=st.error)


                    st.sidebar.subheader("Обучение модели")
                    model_id_load = st.sidebar.text_input("model_id_load", max_chars=20)
                    model_load = {"model_id":model_id_load}
                    if st.sidebar.button("Загрузка"):
                        response = requests.post(f"{API_URL}/load_model", json=model_load)
                        if response.status_code == 200:
                            mess = response.json()[0]["message"]
                            log_and_display(f"Отправленные параметры: {mess}", level="success",
                                            display_func=st.success)
                        else:
                            log_and_display(f"Модели с таким id не существует: {response.status_code}", level="error",
                                            display_func=st.error)

                    if st.sidebar.button("Список моделей"):
                        response = requests.get(f"{API_URL}/list_models")
                        if response.status_code == 200:
                            if  response.json()[0]["models"] != []:
                                for models in response.json():
                                    for el in models["models"]:
                                        model = el["model_id"]
                                        type = el["type"]
                                        hparam = el["hyperparameters"]
                                        if response.status_code == 200:
                                            log_and_display(f"Идентификатор модели: {model}, Тип модели: {type}, Гиперпараметры:{hparam}", level="success",
                                                            display_func=st.success)
                                        else:
                                            log_and_display(f"Нет загруженных моделей: {response.status_code}", level="error",
                                                            display_func=st.error)
                            else:
                                log_and_display(f"Нет загруженных моделей", level="error",
                                                display_func=st.error)
                    if st.sidebar.button("Список датасетов"):
                        response = requests.get(f"{API_URL}/get_datasets")
                        if response.status_code == 200:
                            df_list = response.json()["datasets_nm"]
                            log_and_display(f"Список датасетов: {df_list}", level="success",
                                        display_func=st.success)



                    if st.sidebar.button("Бенчмарк"):
                        # надо вызвать find_context 50 раз на 50 рандомных сэмплах и посчитать время и сохранить его, вывести min, max, mean
                        samples = data["question"].sample(100, random_state=42)
                        times = []
                        for el in samples:
                            start = time.time()
                            context = {"model_id": model_id_load, "question": el}
                            response = requests.post(f"{API_URL}/find_context", json=context)
                            if response.status_code == 200:
                                log_and_display("Тест выполнен успешно", level="success")
                            else:
                                log_and_display(f"Нет модели с таким id: {response.status_code}", level="error",
                                                display_func=st.error)
                            end = time.time()
                            res = end - start
                            times.append(res)
                        mean = sum(times)/len(times)
                        fig, ax = plot_bench(times)
                        st.pyplot(fig)
                        st.success(f"Среднее время извлечения одного ответа: {mean} секунды")


                    if st.sidebar.button("Точность"):
                        params = {"model_id" : model_id_load, "threshold":len(data)}
                        response = requests.post(f"{API_URL}/quality_test", json=params)
                        acc = response.json()["accuracy"]*100
                        if response.status_code == 200:
                            log_and_display(f"Точность: {acc} %", level="success",
                                            display_func=st.success)
                        else:
                            log_and_display(f"Ошибка при запросе API: {response.status_code}", level="error",
                                            display_func=st.error)

                    if st.sidebar.button("Выгрузка моделей"):
                        response = requests.post(f"{API_URL}/unload_model", json={"message": "удаление"})
                        try:
                            for el in response.json():
                                mess = el["message"]
                                if response.status_code == 200:
                                    log_and_display(f"Отправленные параметры: {mess}", level="success",
                                                    display_func=st.success)
                                else:
                                    log_and_display(f"Модели с таким id не существует: {response.status_code}", level="error",
                                                    display_func=st.error)
                        except Exception as e:
                            log_and_display(f"Нет загруженных моделей", level="error", display_func=st.error)


                if st.sidebar.checkbox("Инференс", key="infer", on_change=clear_other_checkboxes, args=("infer",)):
                    text_form("textarea")
                    model_id_inf = st.text_input("model_id", max_chars=20)
                    if st.button('Отправить текст'):
                        if len(st.session_state.textarea):
                            st.success('Текст отправлен в модель')
                        else:
                            st.warning('Поле пустое')
                    if st.session_state.textarea:
                        st.write(st.session_state.textarea)
                        test = st.session_state.textarea
                        context = {"model_id": model_id_inf, "question": test}

                        response = requests.post(f"{API_URL}/find_context", json=context)
                        if response.status_code == 200:
                            request = response.json()[0]["context"]
                            score = response.json()[0]["score"]
                            id = response.json()[0]["point_id"]
                            log_and_display(f"Ответ: {request}", level="success", display_func=st.success)
                            log_and_display(f"Score: {score}, Идентификатор {id}", level="success", display_func=st.warning)
                            log_and_display("Предикт выполнен успешно", level="success")
                        else:
                            log_and_display(f"Нет модели с таким id: {response.status_code}", level="error",
                                            display_func=st.error)
                st.sidebar.subheader("Удаление моделей")

                model_id_remove = st.sidebar.text_input("model_id_remove", max_chars=20)
                if model_id_remove is not None:
                    if st.sidebar.button("Удалить модель"):
                        response = requests.delete(f"{API_URL}/remove/{model_id_remove}")
                        if response.status_code == 200:
                            log_and_display(f"Удалена модель {model_id_remove}", level="success", display_func=st.success)
                        else:
                            log_and_display(f"Модели с таким id не существует: {response.status_code}", level="error",
                                            display_func=st.error)

                if st.sidebar.button("Удалить все модели"):
                    response = requests.delete(f"{API_URL}/remove_all")
                    if response.status_code == 200:
                        log_and_display(f"Удалены все модели", level="success", display_func=st.success)
                    else:
                        log_and_display(f"Нет моделей для удаления: {response.status_code}", level="error",
                                        display_func=st.error)


if __name__ == "__main__":
    main()
