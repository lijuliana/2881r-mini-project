# rh_decomp: what is the reward-hacking probe made of?

CS 2881R project (Niranjan Deshpande). It decomposes the Goodfire-style reward-hacking DoM probe `v_hack` on Qwen3.8-27B,
the one replicated in `../rh_probe`, into candidate components:
- test vocabulary;
- task failure;
- legitimate shortcuts;
- eval awareness;
- grader modeling.

It compares them against unrelated baseline directions, and also against Qwen's emotion directions.

**Where to start:**
- Results: `results/CHECKPOINT_D.md` (main decomposition) and `results/PHASE5_AND_LENS.md` (emotions; logit-lens check). Figures are in `results/figures/`.
- Every choice and its reason: `DECISIONS.md`. Deviations from the original brief: `PLAN.md`. Spending: `COSTS.md`. Full project state: `HANDOFF.md`.

## Layout
- `src/rhdecomp/`: analysis library (stats, directions, geometry, functional, shapley, decision, confounds, emotions; `gen/` for data generation). Tests are in `tests/`.
- `scripts/`: generation (GPT-6 Luna), contamination judging and Phase 2 labels (Luna and Claude Haiku), and the analysis drivers (`geometry_analysis.py`, `phase4.py`, `phase5_emotions.py`, `figures.py`, `lens_discrepancy.py`).
- `gpu/`: the RunPod session scripts. They harvest activations, project rollouts, build emotion vectors and compute the lenses.
- `prompts/`: generation prompts, schemas, rubrics, and the emotion list and topics.
- `data/final/`: the final synthetic pair sets (22 directions) and the Phase 2 consensus labels.
- **Not included** (too large, in `.gitignore`): `data/gpu/`, which holds the activations and per-token projections from the GPU sessions.

## Paths
Scripts locate their inputs relative to this folder:
- `../agastya-repo/rh_probe/...` (Agastya's rollouts, labels and vectors);
- `../olmo-repo/synth/...` (the Claude-written pairs).

In this fork the `rh_probe/` inputs sit at the repository root. Create `agastya-repo` there as a symlink to the root
(`ln -s . agastya-repo`), or check out the repos side by side.

Environment: Python 3.13; `pip install -e .`; API keys in `../.env` (never committed).
