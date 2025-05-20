#!/bin/bash

echo "Запуск формирования эмбеддингов..."
python embeddings_compute.py

# Базовый запуск с настройками по умолчанию
echo "Запуск бенчмарка с настройками по умолчанию..."
python bench.py

# Запуск только с BM25
echo "Запуск бенчмарка только с BM25..."
python bench.py --model-names BM25

echo "Запуск бенчмарка c гибридным поиском и реранкером для коллекции в 11000 элементов"
python bench.py --hybrid 1 --limit 11000

echo "Запуск бенчмарка c гибридным поиском и реранкером для коллекции в 11000 элементов без нового создания и загрузки коллекции"
python bench.py --hybrid 1 --limit 11000 --load 0

# Запуск с конкретными моделями
echo "Запуск бенчмарка с конкретными моделями..."
python bench.py --model-names BM25 jina-embeddings-v2-base-en, snowflake-arctic-embed-s

# Запуск с настройкой параметров HNSW
echo "Запуск бенчмарка с настройкой параметров HNSW..."
python bench.py --hnsw-ef 128 --hnsw-m 32 --ef-construct 400