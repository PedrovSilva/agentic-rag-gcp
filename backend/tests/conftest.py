
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Permite importar a aplicação sem credenciais reais nos testes.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("HF_TOKEN", "test-token")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/rag_test",
)

from app.main import app


@pytest.fixture
def client():
    # Os testes de endpoint não precisam inicializar o banco.
    with patch("app.main.init_db"):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def mock_settings(monkeypatch):
    settings = SimpleNamespace(max_upload_bytes=1024)
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)
    return settings
