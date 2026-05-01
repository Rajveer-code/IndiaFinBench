"""
Tests for demo/app.py API endpoints.
Run from repo root: pytest demo/tests/test_app.py -v
"""
import json
import sys
from pathlib import Path

# Mirror the exact sys.path setup that demo/app.py uses at runtime:
# ROOT first (so `import rag` resolves to /repo/rag/), DEMO second.
_DEMO_DIR = Path(__file__).parent.parent   # demo/
_ROOT_DIR  = _DEMO_DIR.parent              # repo root

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(1, str(_DEMO_DIR))

import pytest

# Importing app triggers init_db() and the RAG warmup background thread.
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


class TestIndex:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_contains_brand(self, client):
        r = client.get("/")
        assert b"IndiaFinBench" in r.data


class TestLeaderboard:
    def test_returns_json_list(self, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert isinstance(data, list)

    def test_has_at_least_9_models(self, client):
        r = client.get("/api/leaderboard")
        data = json.loads(r.data)
        models = [m for m in data if not m.get("is_human")]
        assert len(models) >= 9

    def test_each_model_has_required_keys(self, client):
        r = client.get("/api/leaderboard")
        data = json.loads(r.data)
        for m in data:
            for key in ("label", "overall", "reg", "num", "con", "tmp"):
                assert key in m, f"Missing key '{key}' in model {m.get('label')}"


class TestExample:
    def test_returns_question(self, client):
        r = client.get("/api/example")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "question" in data
        assert "answer" in data
        assert "task_type" in data

    def test_task_filter(self, client):
        r = client.get("/api/example?task=Regulatory+Interpretation")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data.get("task_type") == "Regulatory Interpretation"

    def test_unknown_filter_returns_error(self, client):
        r = client.get("/api/example?task=NotATask&diff=NotADiff")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "error" in data


class TestSubmit:
    def test_returns_issue_url(self, client):
        r = client.post(
            "/api/submit",
            data=json.dumps({"hf_id": "meta-llama/Llama-3.2-3B", "label": "Llama-3.2-3B"}),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "issue_url" in data
        assert "github.com/Rajveer-code/IndiaFinBench/issues/new" in data["issue_url"]
        assert "Llama-3.2-3B" in data["issue_url"]

    def test_hf_id_in_issue_url(self, client):
        r = client.post(
            "/api/submit",
            data=json.dumps({"hf_id": "mistralai/Mistral-7B-v0.3"}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert "issue_url" in data
        assert "mistralai" in data["issue_url"]

    def test_missing_hf_id_returns_400(self, client):
        r = client.post(
            "/api/submit",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
        data = json.loads(r.data)
        assert "error" in data

    def test_no_job_id_in_response(self, client):
        r = client.post(
            "/api/submit",
            data=json.dumps({"hf_id": "some/model"}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert "job_id" not in data


class TestRag:
    def test_empty_query_returns_400(self, client):
        r = client.post(
            "/api/rag",
            data=json.dumps({"query": ""}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_missing_query_key_returns_400(self, client):
        r = client.post(
            "/api/rag",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
