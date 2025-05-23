import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):
    """Функция настроки логера"""
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


if not os.path.exists('logs'):
    os.makedirs('logs')

main_logger = setup_logger('main', 'logs/main.log')
api_logger = setup_logger('api', 'logs/api.log')
qdrant_logger = setup_logger('qdrant', 'logs/qdrant.log')
