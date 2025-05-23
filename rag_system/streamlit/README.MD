# Сервис Streamlit для анализа данных и обучения моделей

## Возможности приложения:

* загрузка датасета и анализ данных
* препроцессинг данных
* конфигурирование и обучение модели
* получение инференса
---------
Для обучения модели, тестирования инференса используется FastAPI.
---------
### Структура проекта:  
```
|-- config.toml - конфигурация темы приложения  
|-- .dockerignore
|-- Dockerfile -- создание образа из наших файлов приложения
|-- eda.py - функции для отрисовки графиков, препроцессинг текстов  
|-- project_logger.py - модуль логирования, установки и функция-обертка  
|-- st_app.py - основной код приложений  
|-- validat_df.py - модуль, отвечающий за валидацию в загружаемого приложение файла  
|-- requirements.txt  - список зависимостей окружения  
|--logs - папка с логами  
      |-- app.log - лог-файл  
```
основной код приложения находится в файле `st_app.py` 

---------
### Запуск для тестирования только frontend части:

- установить зависимости (pip install -r requirements.txt)
- запуск:
 `streamlit run st_app.py --server.runOnSave=false`
---------
### Шаги работы с сервисом:

- на первом шаге работы нужно загрузить датасет (обязательно csv файл), параметры датасета указаны на странице, ожидается получить колонку с контекстами, вопросами и ответами.
- производится валидация загружаемого файла (`validate_df.py`), проверяется размер, тип файла, содержимое
- вывод базовой информации о датасете, пропусках и дубликатах (только информирование)
- вывод EDA графиков (`eda.py`):<!-- {"fold":true} -->
  - распределение по длинам текстов в словах (интерактивный график на `plotly`)
  - вывод наиболее часто встречающихся стоп-слов и не стоп-слов
  - вывод Облака слов
  - вывод t-SNE (данный вывод занимает длительное время, требуется выполнить расчет векторов с помощью `TfidfVectorizer`)
- шаг препроцессинга данных и отправка в модель
  - препроцессинг выполняется функцией `prep`, включает в себя удаление стоп-слов, удаление пунктуации и всех символов, не являющихся буквами и цифрами, приведение к нижнему регистру
  - удаление дубликатов и строк с пустыми значениями
- шаг подготовки к обучению:
  - задаются гиперпараментры для векторизации, тип модели -  `tf-idf`  
  - важно указать идентификатор модели (сервис позволяет загрузить несколько моделей)  
  - при нажатии на кнопку “Сохранить параметры” происходит вызов endpoint : `POST/api/v1/models/fit_save`  
- после сохранения модели на бэке данные векторизуются и помещаются в БД `qdrant`, появляются возможности:  
  - загрузить определенную модель для дальнейшего использования (вызов endpoint `POST/api/v1/models/load_model`)  
  - вывести список имеющихся моделей с гиперпараметрами вызов endpoint `POST/api/v1/models/list_model`)  
  - посмотреть список датасетов вызов endpoint `GET/api/v1/models/get_datasets`)  
  - вывести **Бенчмарк**   
  ⚠️ **Важно:** этот пункт заменяет кривые обучения (не актуально для данной модели)  **Бенчмарк** показывает время извлечения ответа из базы данных, тест производится на 100 случайных примеров (используется `POST/api/v1/models/find_context`), взятых из датасета из колонки с вопросами, построен график, на котором можно увидеть распределение времени извлечения текстов по вопросу  
- точность вычисляется на сервере, производится брутфорс для каждого вопроса извлекается ответ и сравнивается с оригинальным по индексам - вызов endpoint `POST/api/v1/models/quality_test`   

- Инференс позволяет ввести тестовой вопрос в поле и получить ответ - вызов endpoint `POST/api/v1/models/find_context`
- Предусмотрена выгрузка моделей,  удаление модели по идентификатору и удаление всех моделей( вызов endpoints: `POST/api/v1/models/unload_model`, `DELETE/api/v1/models/remove/{model_id}`, `DELETE/api/v1/models/remove_all`)  
---------
### Сравнение моделей  

Сервис позволяет обучить и загрузить несколько моделей, и далее **сравнить** модели.  
Инференс можно проводить с помощью любой из загруженных моделей, для этого нужно указать идентификатор модели.  

В секции `Тестирование моделей` можно указать идентификаторы 2-х моделей и вывести бенчмарк и точность для сравнения результатов моделей.  
Если указать идентификатор одной модели, то выводится результат одной модели без сравнения.

---------
### Логирование    

Лог сохраняется в один файл `app.log`, реализовано через модуль `project_logger.py`, содержит функцию обертку, которая перехватывает и записывает стандартные выводы streamlit (`st.error`, `st.success`, `st.write` и т.д.).  Хранение логов в папке logs, 5 файлов по 10 мб с ротацией.  


---------
### Примечание
Этот проект является частью годового проекта. Данный README описывает лишь часть frontend реализованную на Streamlit

  


