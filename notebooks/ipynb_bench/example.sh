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

# Запуск с конкретными моделями
echo "Запуск бенчмарка с конкретными моделями..."
python bench.py --model-names msmarco-roberta-base-ance-firstp all-MiniLM-L6-v2 msmarco-MiniLM-L-6-v3 BM25

# Запуск с настройкой параметров HNSW
echo "Запуск бенчмарка с настройкой параметров HNSW..."
python bench.py --hnsw-ef 128 --hnsw-m 32 --ef-construct 400

