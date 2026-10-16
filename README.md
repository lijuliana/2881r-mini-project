# Reward-hacking probes on open models (CS 2881R)

Can you catch a model reward-hacking just by reading its activations? Goodfire showed you can, on frontier
models (*"Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations"*,
arXiv 2609.19101). We reproduced it on open models small enough to actually poke at — OLMo 3.1 32B and
Qwen 3.8 27B (plus gpt-oss and GLM-4.5-Air) — and then pushed on three extensions.

The one-line takeaway: the paper's **difference-of-means "reward-hacking direction"** is a **real, readable
thing** — it decodes to cheating and it detects hacking — but it is **not a steering knob**.

## Replication — Goodfire's claims on open models

- **Models reward-hack.** OLMo 3.1 32B is a *competent naive hacker*: it hacks ~46% of impossible coding
  tasks (near the frontier band), but only by hardcoding the tested inputs — it never games the harness the
  sophisticated way bigger models do. → `results/exp1/FINAL_RESULTS.md`
- **Logit lens → cheating words. ✓** The direction decodes to *bypass, fake, cheat, dummy, pretend,
  artificially* at mid-depth, on both OLMo and Qwen — none of which are in the synthetic pairs it's built
  from. Clean replication of the paper's Table 1.
- **The probe detects hacking. ✓** AUROC 0.78–0.84 on OLMo and ~0.93 on Qwen — inside the paper's 0.74–0.97
  band. → `results/probe/RESULTS.md`
- **Probe ≈ LLM monitor? ✗** On OLMo the probe is weaker than a plain Claude monitor at low false-positive
  rates (52% vs 81% TPR at 10% FPR), so the paper's strongest "probe replaces the monitor" claim doesn't
  hold at this scale.
- **Steering causes hacking? ✗** Amplifying the direction (honeypot setup) doesn't induce hacking — it
  degrades the model into gibberish before it ever takes the bait. The one Goodfire result we couldn't
  reproduce at this scale. → `results/probe/INSTRUCT_STEERING.md`

## Extension 1 — Jacobian-lens readout (OLMo)

Is the "cheating" readout real, or a logit-lens artifact? We ran the direction through two more faithful
lenses — a transfer-free Jacobian lens and a borrowed J++ lens. All three agree, and the faithful ones
**sharpen** the reading, pulling **`hardcoded`** to the top at mid-depth — which is exactly the hack OLMo
actually does (100% of its hacks are hardcoding). Native J-lens and J++ agree at cos 0.82–0.86, validating
the borrowed lens. So the direction genuinely encodes cheating. → `results/probe/JLENS.md` (code `probe/jlens/`)

## Extension 2 — What the direction is actually made of (Qwen)

Decomposing `v_hack` shows it's mostly **"take a shortcut" + "notice you're being evaluated"** (and, among
emotions, *low calm* — not desperation). So it's meaningful, not a trivial surface feature — but it's also
tangled up with contradiction / eval-awareness confounds, so it isn't a clean, isolated "hacking" axis.
→ `rh_decomp/` (Niranjan)

## Extension 3 — Across a model zoo (Qwen / gpt-oss / GLM)

The probe + steering pipeline run on three more open models. Detection replicates (Qwen AUROC ~0.93), and
steering — pushing both *toward* the direction and *against* it, including inside the real multi-turn
transcripts on GLM — moves how much the model **talks about** hacking, never whether it **carries it out**,
and breaks the model before it flips. The same readout-not-a-lever story, confirmed across model families.
→ `rh_probe/` (Agastya)

## Layout

- `exp1/`, `probe/` (incl. `probe/jlens/`), `synth/`, `results/` — OLMo work (P(hack), DoM probe, logit +
  J-lens readouts, steering).
- `rh_decomp/` — Qwen direction decomposition (Niranjan).
- `rh_probe/` — multi-model probe + steering pipeline (Agastya).
- `notes/` — paper notes, background, replication plan, protocol.

Large regenerable activation/rollout caches from the Qwen/GLM runs live on the teammates' own forks, not
here, to keep this repo light (see `.gitignore`).

## Running it

Rollouts use [ImpossibleBench](https://github.com/safety-research/impossiblebench) (patched via `patches/`)
against a local vLLM endpoint; the CoT judge and the synthetic-pair generator use the Anthropic API. Each
subproject's `notes/`, `README`, or `HANDOFF` has the specifics.
