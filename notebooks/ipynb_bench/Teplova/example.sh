#!/bin/bash

# Базовый запуск с настройками по умолчанию
echo "Запуск бенчмарка с настройками по умолчанию..."
python client.py

# Запуск только с TF-IDF
echo "Запуск бенчмарка только с TF-IDF..."
python client.py --model-names TF-IDF

# Запуск с конкретными моделями
echo "Запуск бенчмарка с конкретными моделями..."
python client.py --model-names all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2

# Запуск с увеличенным размером выборки
echo "Запуск бенчмарка с увеличенным размером выборки..."
python client.py --limit 500

# Запуск с настройкой параметров HNSW
echo "Запуск бенчмарка с настройкой параметров HNSW..."
python client.py --hnsw-ef 128 --hnsw-m 32 --ef-construct 400

# Запуск с подключением к удаленному Qdrant серверу
# echo "Запуск бенчмарка с подключением к удаленному Qdrant серверу..."
# python client.py --qdrant-host remote-qdrant-server.example.com --qdrant-port 6333 