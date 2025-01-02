import sys
import os

import pytest
from fastapi.testclient import TestClient
from main import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client() -> TestClient:
    """Функция запуска"""
    return TestClient(app)


@pytest.fixture
def test_app():
    """Еще одна функция"""
    return app
