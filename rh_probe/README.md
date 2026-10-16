# Reward-hacking DoM probes on Qwen3.8-27B, gpt-oss-120b and GLM-4.5-Air

Replication of arXiv 2609.19101 (Goodfire) on three open models. It uses Impossible-LiveCodeBench with the strict
prompt and tool harness (conflicting / oneoff / original splits, 103 hard problems, one epoch). The full write-up is
`report/rh_probe_report_filled.html`; open it in a browser.

## Results

| Test | Qwen3.8-27B | gpt-oss-120b | GLM-4.5-Air |
|---|---|---|---|
| T0 hack rate (impossible splits) | 67–70%, but only 2% carried out | 41–71%; 19–42% carried out | 47–56%; 37–46% carried out |
| T1 logit lens | YES: 捷径 "shortcut", 绕过 "bypass" | PARTIAL: " cheating", " unethical" | YES: " pretend", " fraudulent", 造假 "fabricate" |
| T2 held-out AUROC, probe vs monitor | 0.93; probe beats monitor (95% vs 48% TPR) | 0.76; monitor far better | 0.75; monitor better (89% vs 50% TPR at 18% FPR) |
| T2b by stage (considered / carried out) | 0.93 / 0.79 (n = 9) | 0.92 / 0.71 | 0.80 / 0.74 |
| T3 honeypot steering toward v | contemplation 5% → 69% | 12% → 23% | no effect (0/103 → 0/103) |
| T3b steering against v (GLM) | – | – | no reduction in hacking until the model breaks |
| T4 resample at probe peaks | null (0% vs 0.7%) | not run | not run |

**Headline.** The probe works on Qwen, whose hacks are almost all considered in reasoning and almost never carried
out. On the models that carry hacks out (gpt-oss, GLM) it is weak, and a generic text monitor wins. Split by stage,
the probe scores 0.92–0.93 on *considered* hacks but 0.71–0.74 on *carried-out* ones. It detects thinking about
cheating more than doing it.

**T3b, steering GLM against the direction.** We cut each of the 84 GLM rollouts with a carried-out hack just before
that turn and regenerated it twice per condition. Unsteered, GLM re-hacks 32% of the time. At −1× and −1.5× the rate
is unchanged. At −2× it falls to 21%, but 43% of outputs are incoherent; among coherent turns hacking is 35%, the
same as unsteered. −3× is 95% incoherent. A random direction at 3× raises hacking among coherent turns to 53%.

**Capability control.** On the original split (correct tests), the pass rates are Qwen 90%, gpt-oss 78% and
GLM 63%. A judge audit found that 17 of GLM's 65 original-split passes were themselves hacks.

## Setup notes
- **Judge.** GPT-5.6 Sol with the paper's Fig 20 rubric (`rubric_judge_lcb.txt`), 3 runs; only unanimous labels
  are used. **Monitor.** The generic Fig 17 rubric (`rubric_monitor_generic.txt`).
- **Probe and selection.** Probe score = max over a passage's tokens of ⟨v, x_t⟩. Layer, vector and pooling are
  chosen on one hash-split half of the tasks; numbers are reported on the other half, with a task-level bootstrap.
- **Synthetic pairs.** `synth/pairs_full.jsonl` from this repo.
- **GLM serving.** GLM rollouts come from the FP8 checkpoint on vLLM; activations and steering use bf16 in HF. T3b
  steers inside vLLM through `cluster/steer_neg/steerplug`.
- **Inspect gotcha.** Inspect's openai-api provider marks tools `strict` by default. That broke GLM tool calls under
  vLLM, so GLM ran with `--no-strict-tools`.

## Layout
- **Top-level `*.py`** (local machine):
  - Rollout export and judging: `export_rollouts.py`, `judge.py`, `t4_judge.py`, `steer_judge.py`.
  - Analysis: `t0_*`, `t2_*`, `steer_analyze.py`.
  - Report builders: `*_section.py`, `fill_report.py`, `run_final.sh`.
- **`cluster/`** (GPU host). Paths are hard-coded to our cluster.
  - Serving: `serve_*.sh`.
  - Vectors and logit lens: `build_dom.py`.
  - Passage scoring: `t2_acts*.py`.
  - Steering: `t3_steer.py`.
  - Resampling: `t4_resample.py`.
  - Negative steering: `steer_neg/`.
- **`report/`**: the HTML template, the filled report, and the generated fragments.
- **`results/`**:
  - Judge labels per passage: `labels/`.
  - Qwen rollouts, DoM vectors, pair activations and probe scores: `rollouts/`, `dom/qwen/`, `t2/` (see below).
  - T0, T2 and T3 summaries.
  - Raw steering outputs: `t3_raw/`.
  - T4 judged resamples.
  - T3b judged resamples and summaries: `steer_neg/`.

## Qwen data for reproduction on CPU
- **`results/rollouts/qwen_rollouts.jsonl.gz`**: all 271 Qwen3.8-27B rollouts (conflicting 92, original 91,
  oneoff 88). Each row has:
  - `chat`: the full transcript, with reasoning.
  - `passages`: the (step, channel) units that were judged and scored.
  - `tools`: the tool schemas served to the model.
  - `score`, plus task `metadata`.

  Uids match `results/labels/judge_full.jsonl`.
- **`results/dom/qwen/dom_vectors.npz`**: DoM vectors, shape [64 layers, 5120], for span/turn/last × all and each
  hack family. Also `logit_lens.json`.
- **`results/dom/qwen/pair_acts_{span,turn,last}_{hack,honest}.npz`**: per-pair activations for the 135 usable
  synthetic pairs, `acts` [135, 64, 5120] float16, plus `families`. The original `pair_acts.npz` is split one array
  per file to stay under GitHub's 100 MB limit, and the split was checked to be identical. For split-half
  reliability, `dom = acts_hack[idx].mean(0) - acts_honest[idx].mean(0)` over any subset of pairs.
- **`results/t2/qwen_part{1,2,3}/`**: the per-passage probe scores: `scores_0.npz` holds `max`/`mean` arrays
  [passages, 64 layers, 3 keys], and `index_0.jsonl` maps each row to uid / passage / token span. To reproduce the
  held-out AUROC of 0.927 (turn · L40 · max) on CPU:
  ```
  python3 t2_analyze.py --scores results/t2/qwen_part1 results/t2/qwen_part2 results/t2/qwen_part3 \
      --labels results/labels/judge_full.jsonl --out t2_qwen.json
  ```

GLM and gpt-oss raw rollouts and per-token scores are not included; they stay on the cluster.
