"""Replay fixture events and record observed traces, including real process restarts."""
import argparse
import json
import subprocess
import sys
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


def run_worker(events: list[dict], profile: str, storage: Path) -> list[dict]:
    process = subprocess.run(
        [sys.executable, str(ROOT / "helpdesk.py"), "worker", "--profile", profile,
         "--storage", str(storage)],
        input="".join(json.dumps(event) + "\n" for event in events),
        text=True, encoding="utf-8", capture_output=True, check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Scenario worker exited with code {process.returncode}:\n{process.stderr}")
    results = [json.loads(line) for line in process.stdout.splitlines()]
    return [{**event, **result} for event, result in zip(events, results, strict=True)]


def run_scenario(scenario: dict, storage: Path) -> dict:
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

    settings = model_settings()
    log["model"] = settings["model"]
    log["provider"] = "OpenAI-compatible endpoint" if settings["base_url"] else "OpenAI"
    log["events"] = []
    batch = []
    for event in scenario["events"]:
        if event["action"] == "restart_process":
            log["events"].extend(run_worker(batch, profile, storage))
            log["events"].append({**event, "observed": "Worker exited; next worker opens the same storage."})
            batch.clear()
        else:
            batch.append(event)
    log["events"].extend(run_worker(batch, profile, storage))
    return log


def main() -> None:
    load_dotenv()
    scenarios = {scenario["id"]: scenario for scenario in read_fixture("scenarios.json")["scenarios"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", metavar="ID", help="Scenario IDs, or omit to run all.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--direct-tools", action="store_true", help="Also save direct calls without a model.")
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
            })

    with TemporaryDirectory() as shared_storage:
        for scenario_id in args.ids or scenarios:
            scenario = scenarios[scenario_id]
            print(f"Running {scenario_id}", flush=True)
            if scenario["reset_before"]:
                with TemporaryDirectory() as clean_storage:
                    log = run_scenario(scenario, Path(clean_storage))
            else:
                log = run_scenario(scenario, Path(shared_storage))
            write_log(args.output, scenario_id, log)
            print(f"Saved {scenario_id}.json", flush=True)


if __name__ == "__main__":
    main()
