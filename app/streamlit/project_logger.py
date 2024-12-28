
import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    logger = logging.getLogger("StreamlitLogger")
    logger.setLevel(logging.DEBUG)
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
    logger.addHandler(handler)
    return logger


logger = setup_logger()

# Функция-обертка для записи в лог и вывод в Streamlit
def log_and_display(message, level="info", display_func=None):
    if level == "info":
        logger.info(message)
    elif level == "success":
        logger.info("SUCCESS: " + message)
    elif level == "error":
        logger.error("ERROR: " + message)
    else:
        logger.debug(message)

    # отображение в Streamlit
    if display_func:
        display_func(message)
