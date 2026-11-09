"""Replay fixture events and record observed traces, including real process restarts."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from fixtures import ROOT, read_fixture
from helpdesk import model_settings
from helpdesk_tools import check_service_status, find_instructions, prepare_ticket
from routing import route


def write_log(directory: Path, name: str, value: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_worker(events: list[dict], profile: str, storage: Path, offline: bool = False) -> list[dict]:
    command = [sys.executable, str(ROOT / "helpdesk.py"), "worker",
               "--profile", profile, "--storage", str(storage)]
    if offline:
        command.append("--offline")
    process = subprocess.run(
        command,
        input="".join(json.dumps(event) + "\n" for event in events),
        text=True, encoding="utf-8", capture_output=True, check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Scenario worker exited with code {process.returncode}:\n{process.stderr}")
    results = [json.loads(line) for line in process.stdout.splitlines()]
    return [{**event, **result} for event, result in zip(events, results, strict=True)]


def run_scenario(scenario: dict, storage: Path, offline: bool = False) -> dict:
    log = {key: scenario[key] for key in ("id", "title", "part", "status_profile", "reset_before")}
    profile = scenario["status_profile"]
    if scenario["part"] == 1:
        variant = scenario.get("graph_variant", "original")
        log["graph_variant"] = variant
        log["events"] = [
            {**event, **route(event["message"], variant == "with_clarification")}
            for event in scenario["events"]
        ]
        return log

    if offline:
        log.update(model="OfflineScenarioModel", provider="Scripted fake (no LLM)")
    else:
        settings = model_settings()
        log.update(model=settings["model"], provider=model_provider(settings))
    log["events"] = []
    batch = []
    for event in scenario["events"]:
        if event["action"] == "restart_process":
            log["events"].extend(run_worker(batch, profile, storage, offline))
            log["events"].append({**event, "observed": "Worker exited; next worker opens the same storage."})
            batch.clear()
        else:
            batch.append(event)
    log["events"].extend(run_worker(batch, profile, storage, offline))
    return log


def model_provider(settings: dict) -> str:
    if settings["base_url"]:
        return os.environ.get("MODEL_PROVIDER") or "OpenAI-compatible endpoint"
    return "OpenAI"


def archive_drafts(storage: Path, directory: Path) -> list[str]:
    paths = []
    for draft in sorted((storage / "tickets").glob("DEMO-*.json")):
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(draft, directory / draft.name)
        paths.append(f"drafts/{directory.name}/{draft.name}")
    return paths


def main() -> None:
    load_dotenv()
    scenarios = {scenario["id"]: scenario for scenario in read_fixture("scenarios.json")["scenarios"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", metavar="ID", help="Scenario IDs, or omit to run all.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--direct-tools", action="store_true", help="Also save direct calls without a model.")
    parser.add_argument("--offline", action="store_true", help="Rehearse with a scripted fake instead of an LLM.")
    args = parser.parse_args()
    for scenario_id in args.ids:
        if scenario_id not in scenarios:
            parser.error(f"unknown scenario: {scenario_id}")

    if args.direct_tools:
        with TemporaryDirectory() as storage:
            guide = find_instructions("wi-fi")
            status = check_service_status("wifi", "normal")
            draft = prepare_ticket(
                "wifi", "Campus-Demo will not connect.", ["Enabled Wi-Fi"],
                directory=Path(storage) / "tickets", user_id="demo-user-a", thread_id="direct",
                service_status=status["status"], reference_ids=[guide["instruction_id"]],
            )
            write_log(args.output, "direct-tools", {
                "find_instructions": guide, "check_service_status": status, "prepare_ticket": draft,
                "archived_drafts": archive_drafts(Path(storage), args.output / "drafts" / "direct-tools"),
            })

    with ExitStack() as storage_context:
        storage = Path(storage_context.enter_context(TemporaryDirectory()))
        for scenario_id in args.ids or scenarios:
            scenario = scenarios[scenario_id]
            print(f"Running {scenario_id}", flush=True)
            if scenario["reset_before"]:
                storage_context.close()
                storage = Path(storage_context.enter_context(TemporaryDirectory()))
            log = run_scenario(scenario, storage, args.offline)
            drafts = archive_drafts(storage, args.output / "drafts" / scenario_id)
            if drafts:
                log["archived_drafts"] = drafts
            write_log(args.output, scenario_id, log)
            print(f"Saved {scenario_id}.json", flush=True)


if __name__ == "__main__":
    main()
