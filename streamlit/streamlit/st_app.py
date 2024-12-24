import streamlit as st
import pandas as pd

def main():

    # загрузка заголовка приложения
    st.title("Чат-бот на основе RAG")
    st.markdown("""
    ### Возможности приложения
    - загрузка датасета и анализ данных
    - конфигурирование и обучение модели
    - получение инференса
    """)
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

    # вывод начала датафрейма
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Превью - первые 5 строк:")
        st.dataframe(data.head(5))
    else:
        st.write("укажите CSV-файл")


if __name__ == "__main__":
    main()