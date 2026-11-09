"""Scripted fixture rehearsal for environments without a model endpoint."""
import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class OfflineScenarioModel:
    """Exercise the graph with predictable messages; this is not an LLM."""

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        user_messages = [message.content for message in messages if isinstance(message, HumanMessage)]
        request = user_messages[-1].casefold()
        last = messages[-1]
        if isinstance(last, ToolMessage):
            result = json.loads(last.content)
            if last.name == "find_instructions":
                return AIMessage(content="\n".join(result["steps"]))
            if last.name == "prepare_ticket":
                return AIMessage(content=f"Saved local draft {result['file']}. Nothing was sent.")
            if result["status"] == "operational" and "prepare a ticket" in request:
                return self.tool(
                    "prepare_ticket", category="wifi", summary="Campus-Demo still will not connect.",
                    steps_tried=["Enabled Wi-Fi", "Forgot the saved network"], requires_operational=True,
                )
            suffix = {
                "operational": "",
                "outage": " No conditional draft was created.",
                "unknown": " I could not confirm the operating condition. Would you like a draft that states the status is unknown?",
            }
            return AIMessage(content=result["message"] + suffix[result["status"]])
        if "troubleshooting steps" in request:
            return self.tool("find_instructions", topic="wifi")
        if "check" in request and ("status" in request or "whether" in request):
            return self.tool("check_service_status", service="account_recovery" if "recovery" in request else "wifi")
        if request == "prepare a support ticket.":
            return AIMessage(content="What problem are you having and what have you already tried?")
        if "prepare a support ticket." in [message.casefold() for message in user_messages[:-1]]:
            return self.tool(
                "prepare_ticket", category="wifi", summary="Wi-Fi still fails.",
                steps_tried=["Enabled Wi-Fi", "Tried reconnecting"], requires_operational=False,
            )
        kazakh = "Saved language preference: Kazakh." in messages[0].content
        if "which language" in request and kazakh:
            return AIMessage(content="Сіз қазақ тілінде жауап беруімді сұрадыңыз. Wi-Fi және құпиясөзді қалпына келтіру бойынша көмектесе аламын.")
        if "which problem and steps" in request and kazakh and len(user_messages) > 1:
            return AIMessage(content="Wi-Fi қосылмады. Сіз Wi-Fi желісін қосып, қайта қосылып көрдіңіз.")
        if "which steps" in request and len(user_messages) > 1:
            return AIMessage(content="You enabled Wi-Fi, forgot the saved network, and tried reconnecting.")
        if "preferred language or a previous problem" in request:
            return AIMessage(content="This thread has no previous problem and you have no saved language preference.")
        if kazakh:
            return AIMessage(content="Қазақ тілі сақталды. Wi-Fi мәселесі бойынша көмектесе аламын.")
        return AIMessage(content="I have the Wi-Fi issue and the steps you described.")

    @staticmethod
    def tool(name, **args):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"offline-{uuid4().hex}"},
        ])
