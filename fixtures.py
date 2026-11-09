"""Load the fictional helpdesk fixtures independently of the working directory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_fixture(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


HELPDESK = read_fixture("helpdesk.json")
PROFILES = read_fixture("service_status.json")["profiles"]
