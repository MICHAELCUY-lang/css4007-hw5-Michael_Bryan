import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import MAX_TOOL_ROUNDS, HelpdeskSession
from fixtures import ROOT, read_fixture
from helpdesk_tools import check_service_status, find_instructions, prepare_ticket
from routing import route


class FakeModel:
    """Deterministic stand-in for testing the graph; never calls a provider."""

    def __init__(self, callback):
        self.callback = callback

    def bind_tools(self, tools, **kwargs):
        assert {tool["function"]["name"] for tool in tools} == {
            "find_instructions", "check_service_status", "prepare_ticket",
        }
        return self

    def invoke(self, messages):
        return self.callback(messages)


def call(name, args, index):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{index}"}])


@pytest.mark.parametrize(("message", "changed", "category", "node"), [
    ("I cannot connect to Wi-Fi.", False, "wifi", "wifi_help"),
    ("I forgot my PASSWORD.", False, "password", "password_help"),
    ("My laptop screen is broken.", False, "other", "other_help"),
    ("My Wi-Fi password does not work.", False, "wifi", "wifi_help"),
    ("My Wi-Fi password does not work.", True, "both", "clarify_problem"),
])
def test_routing(message, changed, category, node):
    result = route(message, changed)
    first, second = result["trace"]
    assert first == {"node": "read_request", "state": {"message": message, "category": category}}
    assert second == {"node": node, "state": {**first["state"], "reply": result["reply"]}}
    if category in {"wifi", "password"}:
        assert all(step in result["reply"] for step in find_instructions(category)["steps"])
    else:
        assert "?" in result["reply"]


def test_find_instructions():
    assert find_instructions(" Wi-Fi ")["instruction_id"] == "GUIDE-WIFI"
    assert find_instructions("PASSWORD")["instruction_id"] == "GUIDE-PASSWORD"
    assert "error" in find_instructions("printer")


@pytest.mark.parametrize(("profile", "service", "status"), [
    ("normal", "wifi", "operational"),
    ("wifi_outage", "wifi", "outage"),
    ("status_unavailable", "wifi", "unknown"),
    ("recovery_outage", "password", "outage"),
])
def test_check_service_status(profile, service, status):
    result = check_service_status(service, profile)
    assert result["status"] == status
    assert result["service"] == ("account_recovery" if service == "password" else service)


def test_prepare_ticket(tmp_path):
    context = dict(directory=tmp_path, user_id="demo-user-a", thread_id="test")
    assert "error" in prepare_ticket("wifi", "", [], **context)
    assert "error" in prepare_ticket("wifi", "Disconnected", ["   "], **context)
    assert list(tmp_path.iterdir()) == []
    result = prepare_ticket("wifi", "Disconnected", ["Reconnected"], **context)
    ticket = json.loads((tmp_path / Path(result["file"]).name).read_text(encoding="utf-8"))
    assert ticket == result["ticket"]
    assert ticket.keys() == read_fixture("ticket_template.json")["ticket"].keys()
    assert ticket["steps_tried"] == ["Reconnected"]
    assert ticket["service_status"] == "not_checked"
    assert ticket["status"] == "draft"
    assert result["sent"] is False


@pytest.mark.parametrize(("profile", "should_draft"), [
    ("normal", True), ("wifi_outage", False), ("status_unavailable", False),
])
def test_conditional_ticket_uses_tool_result(tmp_path, profile, should_draft):
    # Deliberately try an unsafe call in all profiles to check the graph guard.
    def attempt_ticket(messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return call("check_service_status", {"service": "wifi"}, 1)
        if last.name == "check_service_status":
            return call("prepare_ticket", {
                "category": "wifi", "summary": "Disconnected", "steps_tried": ["Reconnected"],
                "requires_operational": False,
            }, 2)
        result = json.loads(last.content)
        return AIMessage(content="Saved" if "ticket" in result else "Blocked")

    session = HelpdeskSession(FakeModel(attempt_ticket), tmp_path, "demo-user-a", "t", profile)
    with session.connect():
        result = session.turn("Wi-Fi disconnected after reconnecting. Check status. If it is operating, draft a ticket.")
    if profile == "normal":
        assert "no reported outage" in result["reply"]
        assert "Local draft saved as tickets/" in result["reply"]
        assert "Nothing was sent" in result["reply"]
    elif profile == "wifi_outage":
        assert "unavailable across campus" in result["reply"]
        assert "no draft was saved" in result["reply"]
    else:
        assert "status source is unavailable" in result["reply"]
        assert "cannot confirm that it is operating" in result["reply"]
    assert bool(list((tmp_path / "tickets").glob("*.json"))) == should_draft
    assert [entry["node"] for entry in result["trace"]] == [
        "read_memory", "model", "tools", "model", "tools", "model", "report",
    ]


def test_tool_loop_limit(tmp_path):
    def keep_calling(messages):
        return call("check_service_status", {"service": "wifi"}, len(messages))

    session = HelpdeskSession(FakeModel(keep_calling), tmp_path, "demo-user-a", "t", "normal")
    with session.connect():
        result = session.turn("What is the status?")
    assert result["limit_reached"]
    assert sum(entry["node"] == "tools" for entry in result["trace"]) == MAX_TOOL_ROUNDS
    assert result["trace"][-1]["node"] == "limit"


def memory_probe(storage, user, thread, request):
    def read_context(messages):
        # Return the inputs actually received by the fake model.
        return AIMessage(content=json.dumps({
            "system": messages[0].content,
            "user_messages": [message.content for message in messages if isinstance(message, HumanMessage)],
        }))

    session = HelpdeskSession(FakeModel(read_context), Path(storage), user, thread, "normal")
    with session.connect():
        print(session.turn(request)["reply"])


def test_memory_survives_process_restart_and_stays_isolated(tmp_path):
    def launch(user, thread, request):
        process = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'tests'); from test_helpdesk import memory_probe; memory_probe(*sys.argv[1:])",
             str(tmp_path), user, thread, request],
            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
        )
        return json.loads(process.stdout)

    original = "I prefer replies in Kazakh. Wi-Fi failed after reconnecting."
    launch("demo-user-a", "shared-id", original)
    resumed = launch("demo-user-a", "shared-id", "What steps did I try?")
    assert resumed["user_messages"] == [original, "What steps did I try?"]
    assert "Saved language preference: Kazakh." in resumed["system"]

    new_thread = launch("demo-user-a", "new-id", "Which language?")
    assert new_thread["user_messages"] == ["Which language?"]
    assert "Saved language preference: Kazakh." in new_thread["system"]

    new_user = launch("demo-user-b", "shared-id", "Do you know my problem?")
    assert new_user["user_messages"] == ["Do you know my problem?"]
    assert "Saved language preference: none." in new_user["system"]
