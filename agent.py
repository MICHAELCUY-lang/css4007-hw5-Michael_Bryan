"""Parts 2–3: tool execution and persistent conversation/user memory."""
import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field, ValidationError

from fixtures import HELPDESK
from helpdesk_tools import check_service_status, find_instructions, prepare_ticket

MAX_TOOL_ROUNDS = 4

POLICY = """You are the assistant at a fictional university IT helpdesk.
You support Wi-Fi troubleshooting and password recovery.
- Guide request: call find_instructions. Topics: wifi, password.
- Status request: call check_service_status. Services: wifi, account_recovery.
- Conditional ticket: check status first, then draft only if operational.
- Ticket request: if the problem and steps tried are known, call prepare_ticket.
  Otherwise ask for missing facts. Continue the ticket when the user supplies them.
- Recall request: quote the relevant previous user statement to preserve every
  reported step. If no history or preference is available, say so.
- Use tool facts exactly. Never invent symptoms, steps, repair times or secrets.
  A ticket is a local draft. If saved, report its filename.
Answer the user's question directly. Use the saved language if one is set."""

PREFERENCE = re.compile(
    r"\b(?:i prefer replies in|please (?:always )?reply in)"
    r"\s+([A-Za-z]+(?:[ -][A-Za-z]+){0,2})(?=[.!?;,]|\s*$)", re.IGNORECASE,
)
CONDITIONAL_TICKET = re.compile(
    r"\bif\b[^.!?]{0,80}\b(?:operat(?:ing|ional)|working|up)\b", re.IGNORECASE,
)
TICKET_REQUEST = re.compile(
    r"\b(?:prepare|draft|make|create|write|open|need|want|get)\b[^.!?]{0,55}\bticket\b"
    r"|\b(?:ticket draft|support ticket)\s+please\b",
    re.IGNORECASE,
)
RECALL_STEPS = re.compile(r"\bwhich (?:problem and steps|steps) did\b", re.IGNORECASE)
RECALL_LANGUAGE = re.compile(r"\bwhich language did i ask\b", re.IGNORECASE)
SECRET = re.compile(
    r"\b(password|recovery code|api[_ -]?key)\s*[:=]\s*\S+|\bsk-[A-Za-z0-9_-]{12,}",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    return SECRET.sub("[redacted credential]", text)


class InstructionsArgs(BaseModel):
    topic: str = Field(description="wifi or password")


class StatusArgs(BaseModel):
    service: str = Field(description="wifi or account_recovery")


class TicketArgs(BaseModel):
    category: str = Field(description="wifi or password")
    summary: str = Field(description="Problem description based on the user's report")
    steps_tried: list[str] = Field(
        min_length=1, description="One array entry for each action the user already tried. Extract these from the user's words.",
    )
    requires_operational: bool = Field(
        default=False, description="True if the user made the draft conditional on the service operating",
    )


TOOL_ARGUMENTS = {
    "find_instructions": InstructionsArgs,
    "check_service_status": StatusArgs,
    "prepare_ticket": TicketArgs,
}
TOOL_DESCRIPTIONS = {
    "find_instructions": "Retrieve the fictional guide for a supported topic.",
    "check_service_status": "Check a fictional service in the current demo snapshot.",
    "prepare_ticket": "Save one local draft after the user supplies the problem and steps tried.",
}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": name, "description": TOOL_DESCRIPTIONS[name],
        "parameters": schema.model_json_schema(),
    }}
    for name, schema in TOOL_ARGUMENTS.items()
]


class AgentState(MessagesState):
    language: str | None
    tool_rounds: int
    limit_reached: bool


def message_record(message) -> dict:
    record = {"type": message.type, "content": message.content}
    if isinstance(message, AIMessage):
        record["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        record.update(name=message.name, tool_call_id=message.tool_call_id)
    return record


def current_turn(messages: list) -> list:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return []


class HelpdeskSession:
    def __init__(self, model, storage: Path, user_id: str, thread_id: str, profile: str):
        self.storage = storage
        self.user_id = user_id
        self.thread_id = thread_id
        self.profile = profile
        # JSON array encoding keeps arbitrary pairs distinct, even if they contain delimiters.
        self.config = {"configurable": {"thread_id": json.dumps([user_id, thread_id])}}
        self.model = model.bind_tools(TOOL_SCHEMAS, parallel_tool_calls=False)

    @contextmanager
    def connect(self):
        self.storage.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(self.storage / "preferences.sqlite3")) as self.preferences,
            SqliteSaver.from_conn_string(str(self.storage / "conversations.sqlite3")) as checkpointer,
        ):
            self.preferences.execute(
                "CREATE TABLE IF NOT EXISTS preferences (user_id TEXT PRIMARY KEY, language TEXT NOT NULL)"
            )
            graph = StateGraph(AgentState)
            graph.add_node("read_memory", self.read_memory)
            graph.add_node("model", self.call_model)
            graph.add_node("tools", self.execute_tools)
            graph.add_node("report", self.report_verified_result)
            graph.add_node("limit", self.stop_at_limit)
            graph.add_edge(START, "read_memory")
            graph.add_edge("read_memory", "model")
            graph.add_conditional_edges(
                "model", self.next_node, {"tools": "tools", "limit": "limit", "report": "report"}
            )
            graph.add_edge("tools", "model")
            graph.add_edge("report", END)
            graph.add_edge("limit", END)
            self.graph = graph.compile(checkpointer=checkpointer)
            yield self

    def read_memory(self, state: AgentState) -> dict:
        request = current_turn(state["messages"])[0].content
        match = PREFERENCE.search(request)
        if match:
            with self.preferences:
                self.preferences.execute(
                    "INSERT INTO preferences VALUES (?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET language=excluded.language",
                    (self.user_id, match.group(1)),
                )
        row = self.preferences.execute(
            "SELECT language FROM preferences WHERE user_id = ?", (self.user_id,)
        ).fetchone()
        return {
            "language": row[0] if row else None,
            "tool_rounds": 0,
            "limit_reached": False,
        }

    def call_model(self, state: AgentState) -> dict:
        language = state["language"] or "none"
        turn = current_turn(state["messages"])
        prior = state["messages"][:-len(turn)]
        prior_requests = [message.content for message in prior if isinstance(message, HumanMessage)]
        prompt = f"{POLICY}\nSaved language preference: {language}."
        if state["language"]:
            prompt += f" Write your reply in {language}, using that language's normal script."
        if prior_requests:
            prompt += f"\nPrevious statements by this user in this thread: {json.dumps(prior_requests, ensure_ascii=False)}"
        messages = [SystemMessage(content=prompt), *state["messages"]]
        if isinstance(turn[-1], ToolMessage):
            feedback = self.tool_feedback(turn[-1], turn[0].content)
            if feedback:
                payload = json.loads(turn[-1].content)
                payload["runtime_context"] = feedback
                messages[-1] = turn[-1].model_copy(
                    update={"content": json.dumps(payload, ensure_ascii=False)}
                )
        answer = self.model.invoke(messages)
        if isinstance(answer.content, str):
            answer.content = redact(answer.content)
        for call in answer.tool_calls:
            call["args"] = redact_arguments(call["args"])
        return {"messages": [answer]}

    @staticmethod
    def tool_feedback(message: ToolMessage, request: str) -> str:
        result = json.loads(message.content)
        if message.name == "check_service_status" and "status" in result:
            status = result["status"]
            feedback = f"The verified status is {status.upper()}. {result['message']}"
            if CONDITIONAL_TICKET.search(request):
                guidance = {
                    "operational": (
                        "The user's condition is met. They asked for a draft now. Respond with a prepare_ticket "
                        "tool call. Extract the problem and each action already tried from this request: "
                        f"{json.dumps(request, ensure_ascii=False)}"
                    ),
                    "outage": "The condition failed. Explain the outage and say no draft was saved.",
                    "unknown": "The condition is unconfirmed. Explain the unknown status and offer a draft mentioning the uncertainty.",
                }
                feedback += " " + guidance[status]
            return feedback
        if message.name == "prepare_ticket" and "file" in result:
            return f"A local draft was saved as {result['file']}. Report that filename and that it was not sent."
        if message.name == "prepare_ticket" and "error" in result:
            return "No draft was saved. Ask the user what problem they are having and what steps they already tried. Do not assume either."
        return ""

    @staticmethod
    def next_node(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return "report"
        if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
            return "limit"
        return "tools"

    @staticmethod
    def report_verified_result(state: AgentState) -> dict:
        """Use stored facts for replies about drafts, conditional status, and recall."""
        messages = state["messages"]
        turn = current_turn(messages)
        request = turn[0].content
        results = [json.loads(message.content) for message in turn if isinstance(message, ToolMessage)]
        draft = next((result for result in results if "file" in result), None)
        status = next((result for result in reversed(results) if "service" in result), None)
        language = state["language"]
        kazakh = language and language.casefold() == "kazakh"
        reply = None
        if draft:
            filename = draft["file"]
            reply = (
                f"Жергілікті өтінім жобасы сақталды: {filename}. Өтінім жіберілген жоқ."
                if kazakh else f"Local draft saved as {filename}. Nothing was sent."
            )
            if status and status["status"] == "operational" and not kazakh:
                reply = f"{status['message']} {reply}"
        elif status and CONDITIONAL_TICKET.search(request):
            if status["status"] == "outage":
                reply = f"{status['message']} Your condition was not met, so no draft was saved."
            elif status["status"] == "unknown":
                reply = f"{status['message']} I cannot confirm that it is operating. Would you like a draft that records the unknown status?"
        elif RECALL_LANGUAGE.search(request) and kazakh:
            reply = "Сіз қазақ тілінде жауап беруімді сұрадыңыз. Wi-Fi және құпиясөзді қалпына келтіру бойынша көмектесе аламын."
        elif RECALL_STEPS.search(request) and (not language or kazakh):
            prior = messages[:-len(turn)]
            previous = next((message.content for message in reversed(prior) if isinstance(message, HumanMessage)), None)
            if previous:
                preference = PREFERENCE.search(previous)
                if preference:
                    end = preference.end()
                    previous = previous[end:].lstrip(".!?;, ") or previous
                prefix = "Алдыңғы хабарламаңызда былай дедіңіз" if kazakh else "You previously reported"
                reply = f"{prefix}\n\n> {previous}"
        if reply is None:
            return {}
        return {"messages": [AIMessage(id=messages[-1].id, content=reply)]}

    @staticmethod
    def stop_at_limit(state: AgentState) -> dict:
        pending = [
            ToolMessage(
                content='{"error":"Tool-round limit reached."}',
                name=call["name"], tool_call_id=call["id"],
            )
            for call in state["messages"][-1].tool_calls
        ]
        return {
            "messages": [*pending, AIMessage(content="I reached the tool limit for this request. Please try a narrower request.")],
            "limit_reached": True,
        }

    def execute_tools(self, state: AgentState) -> dict:
        results = []
        saved_draft = None
        # A ticket must depend on an earlier model/tool round, even if a provider
        # returns several requests in one batch.
        for call in state["messages"][-1].tool_calls:
            name = call["name"]
            schema = TOOL_ARGUMENTS.get(name)
            if schema is None:
                result = {"error": "Unknown tool."}
            else:
                try:
                    args = schema.model_validate(call["args"])
                except ValidationError:
                    result = {"error": "Missing or invalid tool arguments."}
                else:
                    if name == "prepare_ticket" and saved_draft is not None:
                        result = saved_draft
                    else:
                        result = self.run_tool(name, args, state["messages"])
                        if name == "prepare_ticket" and "ticket" in result:
                            saved_draft = result
            results.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=name, tool_call_id=call["id"],
            ))
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def run_tool(self, name: str, args: BaseModel, messages: list) -> dict:
        if name == "find_instructions":
            return find_instructions(args.topic)
        if name == "check_service_status":
            return check_service_status(args.service, self.profile)

        turn = current_turn(messages)
        recent_requests = [message.content for message in messages if isinstance(message, HumanMessage)][-2:]
        if not any(TICKET_REQUEST.search(request) for request in recent_requests):
            return {"error": "A draft requires an explicit user request."}
        service = HELPDESK["topic_services"].get(args.category.strip().casefold())
        status = "not_checked"
        for message in turn:
            if not isinstance(message, ToolMessage):
                continue
            result = json.loads(message.content)
            if message.name == "check_service_status" and result.get("service") == service:
                status = result["status"]
            if message.name == "prepare_ticket" and "ticket" in result:
                return result

        conditional = args.requires_operational or CONDITIONAL_TICKET.search(turn[0].content)
        if conditional and status != "operational":
            return {"error": "The requested operating condition was not confirmed.", "service_status": status}
        if "[redacted credential]" in args.summary or any(
            "[redacted credential]" in step for step in args.steps_tried
        ):
            return {"error": "Please describe the issue and steps without credentials."}
        guide = find_instructions(args.category)
        reference_ids = [
            guide["instruction_id"] for message in messages
            if isinstance(message, ToolMessage) and message.name == "find_instructions"
            and "instruction_id" in guide
            and json.loads(message.content).get("instruction_id") == guide["instruction_id"]
        ]
        return prepare_ticket(
            args.category, args.summary, args.steps_tried,
            directory=self.storage / "tickets", user_id=self.user_id, thread_id=self.thread_id,
            profile=self.profile, service_status=status, reference_ids=list(dict.fromkeys(reference_ids)),
        )

    def turn(self, request: str) -> dict:
        previous = self.graph.get_state(self.config).values.get("messages", [])
        trace = []
        final = None
        for mode, data in self.graph.stream(
            {"messages": [HumanMessage(content=redact(request))]},
            self.config, stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final = data
                continue
            for node, update in data.items():
                update = dict(update or {})
                if node == "read_memory":
                    update["user_memory"] = {"source": "preferences.sqlite3", "user_id": self.user_id}
                    update["conversation_memory"] = {
                        "source": "conversations.sqlite3",
                        "prior_user_messages": [m.content for m in previous if isinstance(m, HumanMessage)],
                        "prior_message_count": len(previous),
                    }
                if "messages" in update:
                    update["messages"] = [message_record(message) for message in update["messages"]]
                trace.append({"node": node, "update": update})
        return {
            "reply": final["messages"][-1].content,
            "limit_reached": final["limit_reached"],
            "trace": trace,
        }


def redact_arguments(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_arguments(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_arguments(item) for key, item in value.items()}
    return value
