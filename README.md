# Reward-hacking probes on open models (CS 2881R)

Can you catch a model reward-hacking just by reading its activations? Goodfire showed you can, on
frontier models (*"Monitoring and Discovering Reward Hacking with Internal Representations during LLM
Evaluations"*, arXiv 2609.19101). We tried to reproduce it on open models small enough to actually poke
at — OLMo 3.1 32B, Qwen 3.8 27B, GLM-4.5-Air, gpt-oss — and then leaned on the parts that looked shaky.

## What we found

The core object is a **difference-of-means "reward-hacking direction"**: one vector, built from cheap
synthetic hack/honest pairs (never from real rollouts). It turns out to be a **real, readable thing — but
not a steering knob.**

- **It reads as cheating.** Decode the direction with a logit lens and you get *bypass, fake, cheat,
  dummy, pretend, hardcoded* at mid-depth, on every model we tried — even though none of those words are
  in the data it was built from. Clean replication of the paper's Table 1.
- **More faithful lenses agree and sharpen it.** Running the OLMo direction through the Jacobian lens and
  a borrowed J++ lens pulls the explicit tokens to the top and surfaces **`hardcoded`** — which is exactly
  the hack OLMo actually does (100% of its hacks are hardcoding). Three independent lenses agree
  (cos 0.82–0.86), so it isn't a logit-lens artifact. → `results/probe/JLENS.md`
- **It detects hacking.** The probe scores AUROC 0.78–0.84 on OLMo and ~0.93 on Qwen, inside the paper's
  0.74–0.97 band. But on OLMo it's weaker than a plain LLM monitor at low false-positive rates (52% vs
  81% TPR at 10% FPR), so the paper's "probe ≈ monitor" claim doesn't fully hold at this scale.
- **You can't steer with it — the consistent surprise.** Across OLMo (honeypot), Qwen (honeypot), and GLM
  (suppression inside the real transcripts), amplifying the direction changes how much the model *talks
  about* hacking, not whether it *does* it — and pushing hard enough to matter just breaks the model
  first. The direction is a faithful **readout**, not a causal **lever**.
- **What is the direction, really?** Decomposing it (on Qwen) shows it's mostly "take a shortcut" +
  "notice you're being evaluated" — meaningful, not a trivial surface feature — but also tangled up with
  contradiction / eval-awareness confounds, so it isn't a clean "hacking" axis. → `rh_decomp/results/`

And the behavioral baseline underneath all this: **OLMo 3.1 32B is a competent naive hacker.** It hacks
~46% of impossible coding tasks (near the frontier band) but only by hardcoding the tested inputs — it
never games the harness the sophisticated way bigger models do. → `results/exp1/FINAL_RESULTS.md`

## Who did what

- **OLMo 3.1 32B** (core of this repo) — P(hack), the DoM probe, logit + **J-lens / J++** readouts, and
  steering. Code in `exp1/`, `probe/` (incl. `probe/jlens/`), `synth/`; results in `results/`.
- **Direction decomposition on Qwen 3.8 27B** (Niranjan) — what the direction breaks into
  (shortcut / eval-awareness / emotions) and its confounds. `rh_decomp/`.
- **Probe + steering pipeline on Qwen / gpt-oss / GLM** (Agastya) — detection, honeypot steering, and
  suppression-inside-transcripts. `rh_probe/`.

The large activation/rollout caches from the Qwen and GLM runs live on the teammates' own forks, not here,
to keep this repo light (see `.gitignore`).

## Running it

Rollouts use [ImpossibleBench](https://github.com/safety-research/impossiblebench) (patched via `patches/`)
against a local vLLM endpoint; the CoT judge and the synthetic-pair generator use the Anthropic API. Each
subproject's `notes/`, `README`, or `HANDOFF` has the specifics.
