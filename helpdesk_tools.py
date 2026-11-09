"""Reusable local operations used by the graph and by direct callers."""
import json
from pathlib import Path
from uuid import uuid4

from fixtures import HELPDESK, PROFILES, read_fixture


def find_instructions(topic: str) -> dict:
    """Look up the local guide for wifi or password."""
    normalized = topic.strip().casefold()
    for category, aliases in HELPDESK["topic_aliases"].items():
        if normalized in aliases:
            return {"source": "data/helpdesk.json", **HELPDESK["instructions"][category]}
    return {"error": "Unknown topic. Use wifi or password."}


def check_service_status(service: str, profile: str = "normal") -> dict:
    """Read a fictional snapshot for wifi or account_recovery."""
    service = service.strip().casefold()
    service = HELPDESK["topic_services"].get(service, service)
    entry = PROFILES[profile].get(service)
    if entry is None:
        return {"error": "Unknown service. Use wifi or account_recovery."}
    return {
        "source": "data/service_status.json",
        "profile": profile,
        "service": service,
        **entry,
    }


def prepare_ticket(
    category: str,
    summary: str,
    steps_tried: list[str],
    *,
    directory: Path,
    user_id: str,
    thread_id: str,
    profile: str = "normal",
    service_status: str = "not_checked",
    reference_ids: list[str] | None = None,
) -> dict:
    """Save a local draft using the ticket template and the user's reported details."""
    category = category.strip().casefold()
    summary = summary.strip()
    steps_tried = [step.strip() for step in steps_tried if step.strip()]
    if not summary or not steps_tried:
        return {"error": "A problem description and steps already tried are required."}
    if category not in HELPDESK["topic_services"]:
        return {"error": "Unknown ticket category. Use wifi or password."}

    ticket = read_fixture("ticket_template.json")["ticket"]
    ticket_id = f"DEMO-{uuid4().hex}"
    ticket.update(
        ticket_id=ticket_id, user_id=user_id, thread_id=thread_id, category=category,
        summary=summary, steps_tried=steps_tried, status_profile=profile,
        service_status=service_status, reference_ids=reference_ids or [],
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticket_id}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(ticket, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return {"file": f"{directory.name}/{path.name}", "ticket": ticket, "sent": False}
