# AI Development Notes

## AI tools used

* **Claude (Anthropic) via Claude Code** — the coding assistant used to plan, write, run and debug the project from the terminal.
* **Groq-hosted LLM (`openai/gpt-oss-20b` in the verified local run) through LangChain** — the *runtime* AI feature of the application (question → structured intent). It is configurable via `LLM_MODEL`; the code default remains `llama-3.3-70b-versatile`.

## How AI assisted development

1. **Inspection first.** The assistant read the assessment and job description, profiled the CSV with Pandas, and found the three anomalies
   (returned TXN0033, zero quantity TXN0034, missing price TXN0051). It then reverse-engineered the revenue rule from the expected figures:
   `Discount_amount` is a *line total*, and `net = qty × price − discount` reproduces all eight published numbers exactly
   (ignoring discounts gives ₹14,43,700, which does not match).
2. **Architecture proposal before code.** The intent-to-SQL approach, the rule-based fallback and the decision to skip LangGraph were proposed and agreed before implementation.
3. **Incremental phases** (ingestion → analytics → interpreter/safety → API → UI → tests → docs), each run and checked before the next, with a git commit per stage.
4. **Verification, not trust.** Results were compared to the expected figures and to an independent Pandas calculation; the UI was opened in a headless browser and inspected;
   **mutation checks** were run on the tests (see below).

## Problems encountered and how they were debugged

| # | Problem | How it was found | Root cause and fix |
|---|---|---|---|
| 1 | "Total revenue for Monitor in the South region" returned a *breakdown by region* | Ran a battery of 20+ realistic questions through the service | The parser treated the word "region" + "revenue" as a breakdown request. Fix: grouping now needs an explicit "by/per/each/highest" cue. |
| 2 | "Online vs retail revenue by channel" was refused as an invalid month comparison | Same battery | "vs" was treated as a month-comparison keyword. Fix: comparison needs two months; otherwise the question flows on to normal parsing. |
| 3 | "Total revenue for Banana" silently returned the whole-company total | Same battery | Rule parser ignored unknown entities. Fix: heuristic that rejects unknown capitalised names; with the LLM path the same case is caught by filter validation against database values. |
| 4 | "And for retail?" lost the previous analysis | Follow-up test conversation | Parser had no intent for a filter-only fragment. Fix: the parser returns the filters; the service re-uses the previous intent with the new filter. |
| 5 | A chained-comparison bug (`a in {...}.get(x) == y`) in the follow-up code | Caught on re-reading the code *before* running it | Rewrote it as a named mapping (`INTENT_DIMENSION`) and a simple equality test. |
| 6 | After tightening a test, the injection example was refused as "unknown entity 'OR'" | New, stricter assertion failed | The heuristic mistook the capitalised connective for a product name. Fix: connectives added to its allowlist. |
| 7 | Disabling the "SELECT only" rule in the SQL guard broke no test | Deliberate mutation of the code to check the tests can fail | Other rules overlapped, so no test isolated it. Fix: added `EXPLAIN …` and CTE cases that only that rule rejects. |
| 8 | Default Streamlit "primary" button was red, a KPI tile truncated "Monitor / South" | Screenshot review | Added a theme colour, split the KPI into two tiles. |

**Debugging approach:** reproduce with the smallest input → read the exact response (intent, filters, SQL, parameters are all returned by `/ask`) →
find the root cause in the pipeline stage that produced it → fix at that stage → add a test → re-run the whole suite and the mutation checks.

### Mutation checks on the tests (all caught)

| Deliberate bug | Result |
|---|---|
| Ignore `Discount_amount` | 15 tests failed |
| Treat returned rows as revenue | 13 tests failed |
| Remove SELECT-only check | 2 tests failed (after fix #7) |
| String-interpolate filter values into SQL | 3 tests failed |

## Verification status

* **Live Groq integration: verified locally.** The running application reported `llm_enabled: true` and `llm_model: openai/gpt-oss-20b` from `/health`. A real `/ask` request for total revenue returned `interpreter: "llm"` and the expected result (₹14,39,200 across 49 revenue-eligible transactions).
* **Follow-up context: verified in the UI.** After asking for Monitor revenue in the South region, the follow-up monthly-trend question retained both filters and returned ₹94,500 across March and April 2026.
* **Automated tests: 80 passed.** Run `python3 -m pytest -q`; tests use fake LangChain runnables so the suite remains deterministic and does not require a live API key.
* The live model is external and can still be unavailable because of API quota, network, or provider errors; the rule-based fallback is designed to handle those failures.

## What I would improve with another day

1. Run an evaluation set (50+ paraphrased questions with expected intents) against the live model and measure accuracy and fallback rate.
2. Let the LLM ask a clarifying question when a request is ambiguous (e.g. missing second month) — a natural fit for a small **LangGraph** with a clarify node.
3. Persist sessions in Redis/SQLite; add authentication and rate limiting.
4. Add more intents (average order value, top-N, quarter-over-quarter, discount impact) using the same template mechanism.
5. Move to PostgreSQL and containerise (Dockerfile, docker-compose), add Ruff and mypy to CI.
6. Add a RAG layer for definitions/policies ("what counts as revenue?") using retrieval over documentation, keeping numbers on the SQL path.
