from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_root_redirects_to_static_app() -> None:
    client = TestClient(create_app(), follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/app/"


def test_static_app_serves_index_html() -> None:
    client = TestClient(create_app())

    response = client.get("/app/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Production RAG Assistant" in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="document-form"' in response.text
    assert 'id="reindex-dry-run"' in response.text


def test_static_app_serves_assets() -> None:
    client = TestClient(create_app())

    script_response = client.get("/app/app.js")
    style_response = client.get("/app/app.css")

    assert script_response.status_code == 200
    assert "const state" in script_response.text
    assert "uploadDocument" in script_response.text
    assert "reindexDocuments" in script_response.text
    assert style_response.status_code == 200
    assert ".chat-panel" in style_response.text
    assert ".knowledge-panel" in style_response.text
