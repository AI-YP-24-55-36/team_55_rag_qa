import logging
import datetime
import sys
from pathlib import Path
from log_output import Tee
from load_config import load_config

def setup_paths():
    """Загружает пути из конфигурации и возвращает базовые директории"""
    config = load_config()
    base_dir = Path(config["paths"]["base_dir"])
    logs_dir = base_dir / config["paths"]["logs_dir"]
    graphs_dir = base_dir / config["paths"]["graphs_dir"]
    output_dir = base_dir / config["paths"]["output_dir"]
    return base_dir, logs_dir, graphs_dir, output_dir


def setup_logging(logs_dir, output_dir, to_file=True, logger_name='bench'):
    """Настраивает логгер и перенаправляет stdout в файл"""
    # Имя лог-файла с таймстампом
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Дублирование stdout в файл
    if to_file:
        log_file_path = output_dir / f"log_{timestamp}.txt"
        sys.stdout = Tee(str(log_file_path))

    # Настройка логгера
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(logs_dir / "bench.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger