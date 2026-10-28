import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import auth as auth_module
from app.nodes import agent as agent_module
from app.nodes import intent as intent_module
from app.nodes import rewrite as rewrite_module
from app.nodes.intent import IntentDecision
from app.nodes.rewrite import RewriteDecision


class FakeAgentModel:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, _messages):
        if len(self._responses) > 1:
            template = self._responses.pop(0)
        else:
            template = self._responses[0]
        return AIMessage(
            content=template.content,
            tool_calls=[dict(call) for call in template.tool_calls],
        )


@pytest.fixture(autouse=True)
def stub_language_models(monkeypatch):
    def fake_classify_intent_l2(query, messages):
        return IntentDecision(
            intent="other",
            confidence=0.0,
        )

    def fake_rewrite_query_l2(query, messages):
        return RewriteDecision(rewritten_query=query)

    monkeypatch.setattr(
        intent_module,
        "classify_intent_l2",
        fake_classify_intent_l2,
    )
    monkeypatch.setattr(
        rewrite_module,
        "rewrite_query_l2",
        fake_rewrite_query_l2,
    )
    monkeypatch.setattr(
        agent_module,
        "get_agent_runnable",
        lambda: FakeAgentModel([AIMessage(content="测试回复")]),
    )


@pytest.fixture
def install_fake_agent(monkeypatch):
    def install(responses):
        model = FakeAgentModel(responses)
        monkeypatch.setattr(agent_module, "get_agent_runnable", lambda: model)
        return model

    return install


@pytest.fixture(autouse=True)
def isolate_user_store(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_module, "USERS_DB_PATH", tmp_path / "users.db")
    auth_module.get_user_store.cache_clear()
    yield
    auth_module.get_user_store.cache_clear()


@pytest.fixture
def auth_headers():
    """注册用户并登录,返回 Bearer 请求头;按注册顺序分配 user-001/user-002…"""

    def make(
        client: TestClient,
        username: str,
        password: str = "secret123",
        role: str = "customer",
    ) -> dict[str, str]:
        register = client.post(
            "/auth/register",
            json={"username": username, "password": password, "role": role},
        )
        assert register.status_code == 201
        login = client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        assert login.status_code == 200
        return {"Authorization": f"Bearer {login.json()['token']}"}

    return make
