# CSS-4007 HW5 — University IT helpdesk

This terminal application demonstrates deterministic routing, model-selected
tools, and persistent memory with LangGraph. Its university, status snapshots,
users, and ticket drafts are fictional. See [SUBMISSION.md](SUBMISSION.md) for
the graphs, design decisions, and recorded results.

## Install

The implementation was tested with Python 3.12.14. From the repository root
in PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell blocks
activation, use `.\.venv\Scripts\python.exe` in place of `python` below.

## Part 1

No API key is needed. The CLI prints the reply and the complete state after
each node.

```powershell
python helpdesk.py route "I cannot connect to Wi-Fi."
python helpdesk.py route "My Wi-Fi password does not work." --variant original
python helpdesk.py route "My Wi-Fi password does not work." --variant with_clarification
python run_scenarios.py S01 S02 S03 S04 S05
```

## Parts 2–3

Copy the example settings with `Copy-Item .env.example .env` on PowerShell
or `cp .env.example .env` on macOS/Linux.

- For OpenAI, set `OPENAI_API_KEY` and `MODEL_NAME` in `.env`. Choose an
  available Chat Completions model with function calling enabled.
- For a local tool-capable model, run an OpenAI-compatible server. Set
  `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` and set `MODEL_NAME` to its model
  alias. Leave `OPENAI_API_KEY` blank for a local server that accepts unauthenticated
  requests. See [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
  for server setup with `--jinja`.

```powershell
python helpdesk.py chat --user demo-user-a --thread wifi-1 --profile normal --trace
```

Enter these messages separately:

1. `Show me the Wi-Fi troubleshooting steps.`
2. `I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.`

Enter `/exit` to close the process. Repeat the same command to resume
`wifi-1`, or change `--thread` to start a new conversation. Select one of
`normal`, `wifi_outage`, `recovery_outage`, or `status_unavailable` when
starting a session. Use invented identifiers and descriptions without credentials.

## Replay the assignment

```powershell
python run_scenarios.py --direct-tools --output local-storage/replay
python run_scenarios.py S06 S07 S08 --output local-storage/status-replay
python -m pytest -q
```

The first command runs S01–S14 and the three direct tool calls. The second
replays the Wi-Fi status cases. Both use the actual configured model for
S06–S14. Each run creates isolated temporary storage, preserves it across a
fixture restart, and records worker process IDs. Logs include node updates,
tool requests and results, memory reads, replies, and the loop limit flag.
The runner copies fictional drafts into the specified output folder before
discarding temporary databases. Use a fresh output directory for a fresh replay.
Pytest uses a deterministic fake model and needs no key.

## Storage and reset

Interactive runs use `local-storage/` by default. Set
`HELPDESK_STORAGE_DIR` or pass `--storage` to choose another directory.

| Path | Contents |
| --- | --- |
| `conversations.sqlite3` | LangGraph checkpoints scoped to the user and thread |
| `preferences.sqlite3` | Saved language keyed by user |
| `tickets/` | Local JSON drafts with unique identifiers |

To reset, exit the program and delete its storage directory, or start the
program with a fresh `--storage` path. The default directory and database
files are ignored by Git.

## Publish the assignment

Fork [the starter](https://github.com/ardakshalkar/css4007-hw5) to a public
repository on your GitHub account. From this completed repository, add your
fork as a remote and push the staged commits:

```powershell
git remote add fork https://github.com/YOUR-USERNAME/css4007-hw5.git
git push -u fork main
```

Check the **Actions** tab on your fork for the test run. Submit your fork's
repository link through the HW5 LMS destination supplied by your instructor.
