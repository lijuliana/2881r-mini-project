# Replicating reward-hacking probes on open models (CS2881R mini-project)

Replication + extension of Goodfire's *"Monitoring and Discovering Reward Hacking with Internal
Representations during LLM Evaluations"* (arXiv 2609.19101) on small open models (OLMo 3.1 32B Think,
later a second model), instead of the paper's 1T+ frontier models.

## Status
- **Phase 0 — P(reward hack): DONE.** Measured how often OLMo 3.1 32B Think reward-hacks on
  Impossible-LiveCodeBench. Result: it is a *competent naive hacker* — ~46% hack propensity on impossible
  tasks (near the frontier band), ~29% successful hacks where hardcoding suffices, but ~0% where the hack
  needs sophistication it lacks. See `results/exp1/FINAL_RESULTS.md`.
- **Phase 1 — DoM probe replication: DONE (OLMo).** Built a difference-of-means reward-hacking direction
  from 211 synthetic contrast pairs. Results in `results/probe/RESULTS.md`:
  - **Logit-lens: replicates** — the vector decodes to cheating words (bypass/fake/cheat/dummy/pretend).
  - **Detection: core claim replicates** — AUROC 0.78–0.84 (in the paper's 0.74–0.97 range); but weaker
    than our LLM monitor at low FPR (52% vs 81% TPR @ 10% FPR), unlike the paper's probe≈monitor.
  - **Honeypot steering: does not cleanly transfer** — OLMo Think reasons too long to reach code within a
    feasible generation budget, so the paper's bait-uptake protocol isn't measurable here.
  Plan in `notes/replication_plan.md`.
- **Three-model replication (Qwen3.8-27B, gpt-oss-120b, GLM-4.5-Air): DONE.** Full T0–T4 replication with a
  stage split (the probe scores 0.92–0.93 on considered hacks vs 0.71–0.74 on carried-out ones), a capability
  control, and negative steering on GLM (no reduction in hacking until the model breaks). See `rh_probe/README.md`
  and `rh_probe/report/rh_probe_report_filled.html`.

## Layout
- `notes/` — `paper_notes.md` (method we replicate), `background_research.md` (models/benchmarks/prior work),
  `replication_plan.md` (Phase-1 plan), `exp1_impossible_lcb_protocol.md` (Phase-0 methods + findings).
- `exp1/` — Phase-0 code: run rollouts (`run_pilot.py`), CoT hack judge (`judge_cot.py`,
  `judge_consensus.py`), analysis (`analyze.py`), serving (`vm_serve_and_harness.sh`, `smoke_test.py`).
- `synth/` — `generate_pairs.py`: Claude-generated hack/honest minimal pairs for building DoM vectors.
- `results/exp1/` — `transcripts_full.jsonl` (150 rollouts), `labels_consensus.jsonl` (3× judge labels),
  `FINAL_RESULTS.md`.
- `patches/` — audit fixes for ImpossibleBench (exit-code & find_code bugs, `exclude_broken`).

## Setup
Rollouts use [ImpossibleBench](https://github.com/safety-research/impossiblebench) (clone into `vendor/`,
apply `patches/impossiblebench-lcb-fixes.patch`) served against a local vLLM OLMo endpoint. The judge and
synthetic-data generator use the Anthropic API. See `notes/` for specifics.
