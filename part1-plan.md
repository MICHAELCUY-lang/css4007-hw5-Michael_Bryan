# Part 1 — routing plan

The original graph starts at `read_request`, which reads the original message and
updates only `category`. The conditional edge reads that category to choose
`wifi_help`, `password_help`, or `other_help`. Each help node writes `reply`.
The two topic-specific nodes use the corresponding local guide. LangGraph
merges each partial update into the current state before passing the state
onward. The conditional edge chooses an edge without writing to state.
`START` marks entry and `END` marks completion. Compilation validates the
structure and returns a runnable graph.

Predictions before running:

| Scenario | Prediction |
| --- | --- |
| S01 | read_request → wifi_help |
| S02 | read_request → password_help |
| S03 | read_request → other_help |
| S04 | read_request → wifi_help because Wi-Fi is checked first |
| S05 | read_request → clarify_problem after adding the combined-topic check |
