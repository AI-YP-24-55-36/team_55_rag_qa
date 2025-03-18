from pathlib import Path
import yaml


CONFIG_FILE = "config.yml"
def load_config():
    default_config = {
        "paths": {
            "base_dir": ".",
            "logs_dir": "logs",
            "graphs_dir": "logs/graphs",
            "embeddings_dir": "embeddings",
            "output_dir": "output"
        }
    }

    # Проверяем существование файла конфигурации
    if not Path(CONFIG_FILE).exists():
        # Создаем конфигурацию по умолчанию
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

    # Загружаем конфигурацию
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


# Загружаем конфигурацию
config = load_config()

# Создаем объекты Path из конфигурации
BASE_DIR = Path(config["paths"]["base_dir"])
LOGS_DIR = BASE_DIR / config["paths"]["logs_dir"]
GRAPHS_DIR = BASE_DIR / config["paths"]["graphs_dir"]
EMBEDDINGS_DIR = BASE_DIR / config["paths"]["embeddings_dir"]

# Создание всех необходимых директорий
LOGS_DIR.mkdir(exist_ok=True, parents=True)
GRAPHS_DIR.mkdir(exist_ok=True, parents=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)
