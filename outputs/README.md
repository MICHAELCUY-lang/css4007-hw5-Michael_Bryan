# Recorded runs

Each `S*.json` file records a fixture scenario run by `run_scenarios.py`.
S06–S14 were recorded with Qwen3-0.6B-Q4_K_M through a local llama.cpp
b12224 endpoint with reasoning and tool calling enabled.
`direct-tools.json` records calls made without a model. The logs show fictional
inputs, execution paths, tool decisions, and persisted context.
Archived drafts live under `drafts/`; their names match the files returned by
the draft tool. The temporary checkpoint databases are discarded.
See [SUBMISSION.md](../SUBMISSION.md) for the scenario-by-scenario analysis.
