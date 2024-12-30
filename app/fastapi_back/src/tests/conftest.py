import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture
def test_app():
    return app