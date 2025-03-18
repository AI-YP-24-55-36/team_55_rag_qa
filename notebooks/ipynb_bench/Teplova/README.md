# Бенчмарк RAG системы

Этот инструмент позволяет сравнивать производительность различных моделей эмбеддингов и алгоритмов поиска для Retrieval-Augmented Generation (RAG) систем.

Демо-скринкаст работы скрипта `client.py`  
https://disk.yandex.ru/i/bn1o9ytBIm5NoA

## Возможности

- Сравнение различных моделей эмбеддингов (SentenceTransformer)
- Сравнение с BM25
- Сравнение с классическим TF-IDF подходом
- Оценка скорости и точности поиска
- Визуализация результатов в виде графиков
- Поддержка различных алгоритмов поиска (Exact Search, HNSW (Hierarchical Navigable Small World))

## Требования

- Python 3.10+
- Qdrant (запущенный локально или удаленно)
- Необходимые библиотеки (см. `requirements.txt`)

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### Базовый запуск

```bash
python client.py
```

### Параметры командной строки

```bash
python client.py --model-names all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2 TF-IDF --limit 100
python client.py --model-names all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2 BM25 --limit 100
```
В результате получаем два графика для одного бенча:

- сравнение по скорости поиска по трем методам поиска:
<img width="1125" alt="image" src="https://github.com/user-attachments/assets/d2879a56-4724-4865-aad9-bbb16f378c7a" />

<img width="1161" alt="image" src="https://github.com/user-attachments/assets/ba30dab0-9bf6-42f9-a36d-9aa281b96f8f" />


- сравнение по точности поиска между разными моделями конвертации текстов в эмбеддинги и методами поиска
<img width="1124" alt="image" src="https://github.com/user-attachments/assets/e8ca95d4-d1ec-48ad-b230-ef24c7f6056c" />

<img width="1165" alt="image" src="https://github.com/user-attachments/assets/29bcb58b-1f2a-479b-bbd3-44b5a90e7fe8" />

результаты сохраняются в файл:

﻿﻿<img width="711" alt="image" src="https://github.com/user-attachments/assets/d78be4fb-8002-49f9-8229-45c11bb48df1" />




### Основные параметры

| Параметр | Описание | Значение по умолчанию |
|----------|----------|------------------------|
| `--model-names` | Список моделей для сравнения | `all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2 TF-IDF` |
| `--limit` | Максимальное количество записей для использования | `100` |
| `--qdrant-host` | Хост Qdrant сервера | `localhost` |
| `--qdrant-port` | Порт Qdrant сервера | `6333` |
| `--collection-name` | Название коллекции в Qdrant | `rag` |

### Параметры HNSW

| Параметр | Описание | Значение по умолчанию |
|----------|----------|------------------------|
| `--hnsw-ef` | Параметр ef для HNSW | `16` |
| `--hnsw-m` | Параметр m для HNSW (количество соседей) | `16` |
| `--ef-construct` | Параметр ef_construct для HNSW | `200` |

## Примеры использования

Для запуска предустановленного пайплайна обучения набора моделей используется баш-скрипт `example.sh`,
он позволяет запустить и сравнить сразу 5 вариантов работы модели с разной векторизацией и разными параметрами `HNSW`  


### Сравнение конкретных моделей

```bash
python client.py --model-names all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2
```

### Только TF-IDF анализ

```bash
python client.py --model-names TF-IDF
```

### Увеличение размера тестовой выборки

```bash
python client.py --limit 1000
```

### Настройка параметров HNSW

```bash
python client.py --hnsw-ef 128 --hnsw-m 32 --ef-construct 400
```

## Результаты

Результаты бенчмарка сохраняются в следующих местах:

- Логи: `./logs/client.log` и `./logs/bench.log`
- Графики: `./logs/graphs/`

## Структура проекта

- `client.py` - основной скрипт для запуска бенчмарка
- `bench.py` - функции для оценки производительности и визуализации
- `read_data_from_csv.py` - функции для чтения данных

## Примечания

- Для корректной работы необходим запущенный сервер Qdrant
- Первый запуск может занять больше времени из-за загрузки моделей
- Для больших наборов данных рекомендуется увеличить значение `--batch-size` 

## Выводы:

1. Exact_Search работает медленнее всего
2. Sparce вектора извлекаются быстрее, чем dense
3. Лучшая точность у Sparce векторов (bm25) 
4. Метод поиска в нашем случае не влияет на точность
5. Были испробованы разные значени ef_construct - параметра качества при построении индекса, при увеличении значения, время извлечения текста увеличивается
6. При увеличении гиперпараметра m - количество связей для каждой вершины, время извлечени контекста увеличивается
7. Изменение гиперпараметра hnsw_ef в нашем случае не сказалось на качестве (вероятно связано со структурой датасета, он довольно подогнанный под задачу)
8. Так как данных мало, то нужно изменять параметр indexing_threshold, чтобы запустить построение графа в HNSW
`optimizers_config=models.OptimizersConfigDiff(indexing_threshold=50)`

