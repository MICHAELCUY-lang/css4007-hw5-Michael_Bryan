"""Deterministic Part 1 graphs."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from fixtures import HELPDESK


class RoutingState(TypedDict, total=False):
    message: str
    category: str
    reply: str


def wifi_help(state: RoutingState) -> dict:
    return instructions("wifi")


def password_help(state: RoutingState) -> dict:
    return instructions("password")


def instructions(topic: str) -> dict:
    guide = HELPDESK["instructions"][topic]
    reply = "\n".join([guide["title"], *(
        f"{number}. {step}" for number, step in enumerate(guide["steps"], 1)
    )])
    return {"reply": reply}


def other_help(state: RoutingState) -> dict:
    return {"reply": "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state: RoutingState) -> dict:
    return {"reply": "Which would you like to solve first: connecting to Wi-Fi or recovering your account password?"}


def build_routing_graph(with_clarification: bool = False):
    def read_request(state: RoutingState) -> dict:
        message = state["message"].casefold()
        wifi = "wifi" in message or "wi-fi" in message
        password = "password" in message
        if with_clarification and wifi and password:
            return {"category": "both"}
        if wifi:
            return {"category": "wifi"}
        if password:
            return {"category": "password"}
        return {"category": "other"}

    routes = {"wifi": wifi_help, "password": password_help, "other": other_help}
    if with_clarification:
        routes["both"] = clarify_problem

    graph = StateGraph(RoutingState)
    graph.add_node("read_request", read_request)
    graph.add_edge(START, "read_request")
    for node in routes.values():
        graph.add_node(node)
        graph.add_edge(node.__name__, END)
    graph.add_conditional_edges(
        "read_request", lambda state: state["category"],
        {category: node.__name__ for category, node in routes.items()},
    )
    return graph.compile()


def route(message: str, with_clarification: bool = False) -> dict:
    graph = build_routing_graph(with_clarification)
    trace = []
    state = {"message": message}
    for updates in graph.stream(state, stream_mode="updates"):
        for node, update in updates.items():
            state = {**state, **update}
            trace.append({"node": node, "state": state})
    return {"trace": trace, "reply": state["reply"]}
