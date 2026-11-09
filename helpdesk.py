"""Terminal entry point and JSON-lines protocol for the scenario runner."""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from fixtures import PROFILES
from routing import route


def model_settings() -> dict:
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("Set MODEL_NAME in .env before running the model graph.")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    if base_url is None and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env or configure a local OPENAI_BASE_URL.")
    return {"model": model, "base_url": base_url}


def create_model(settings: dict):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        **settings,
        api_key=os.environ.get("OPENAI_API_KEY") or "local",
        temperature=0,
        timeout=120,
        max_retries=1,
        use_responses_api=False,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    part1 = commands.add_parser("route", help="Run the deterministic Part 1 graph.")
    part1.add_argument("message")
    part1.add_argument("--variant", choices=("original", "with_clarification"), default="original")

    for command in ("chat", "worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--profile", choices=PROFILES, default="normal")
        sub.add_argument("--storage", type=Path, default=Path(
            os.environ.get("HELPDESK_STORAGE_DIR", "local-storage")
        ))
        if command == "chat":
            sub.add_argument("--user", required=True)
            sub.add_argument("--thread", required=True)
            sub.add_argument("--trace", action="store_true")
        else:
            sub.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    return root


def main() -> None:
    load_dotenv()
    options = parser()
    args = options.parse_args()
    if args.command == "route":
        result = route(args.message, args.variant == "with_clarification")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return

    from agent import HelpdeskSession

    if args.command == "worker" and args.offline:
        from offline_model import OfflineScenarioModel

        model = OfflineScenarioModel()
    else:
        try:
            settings = model_settings()
        except ValueError as error:
            options.error(str(error))
        model = create_model(settings)

    if args.command == "worker":
        for line in sys.stdin:
            event = json.loads(line)
            session = HelpdeskSession(
                model, args.storage, event["user_id"], event["thread_id"], args.profile,
            )
            with session.connect():
                result = session.turn(event["message"])
            print(json.dumps({"pid": os.getpid(), **result}, ensure_ascii=True), flush=True)
        return

    session = HelpdeskSession(model, args.storage, args.user, args.thread, args.profile)
    with session.connect():
        while True:
            try:
                request = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if request == "/exit":
                break
            if request:
                result = session.turn(request)
                print(f"Helpdesk> {result['reply']}")
                if args.trace:
                    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
