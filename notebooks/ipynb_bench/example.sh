#!/bin/bash

# Базовый запуск с настройками по умолчанию
echo "Запуск бенчмарка с настройками по умолчанию..."
python client.py

# Запуск только с BM25
echo "Запуск бенчмарка только с BM25..."
python client.py --model-names BM25

echo "Запуск бенчмарка c гибридным поиском и реранкером для коллекции в 1000 элементов"
python client.py --hybrid 1 --limit 1000

# Запуск с конкретными моделями
echo "Запуск бенчмарка с конкретными моделями..."
python client.py --model-names all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2

# Запуск с увеличенным размером выборки
echo "Запуск бенчмарка с увеличенным размером выборки..."
python client.py --limit 500

# Запуск с настройкой параметров HNSW
echo "Запуск бенчмарка с настройкой параметров HNSW..."
python client.py --hnsw-ef 128 --hnsw-m 32 --ef-construct 400

