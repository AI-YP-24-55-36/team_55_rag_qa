import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    """Установка настроек логера"""
    logger_ = logging.getLogger("StreamlitLogger")
    logger_.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger_.addHandler(handler)
    return logger_


def log_and_display(message, level="info", display_func=None):
    """Функция обертка для записи в лог-файл и вывод в Streamlit"""
    if level == "info":
        logger.info(message)
    elif level == "success":
        logger.info("SUCCESS: %s", message)  # Используем ленивое форматирование
    elif level == "error":
        logger.error("ERROR: %s", message)
    else:
        logger.debug(message)
    # отображение в Streamlit
    if display_func:
        display_func(message)


logger = setup_logger()
