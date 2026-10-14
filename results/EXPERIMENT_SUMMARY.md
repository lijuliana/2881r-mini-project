# OLMo 3.1 32B results (reward-hacking replication)

Our contribution: OLMo 3.1 32B Think (main) + OLMo 3.1 32B Instruct (for the steering experiment).
Benchmark: Impossible-LiveCodeBench (same family the paper used). Judge: Sonnet 4.5, 3x consensus.
150 rollouts = 25 tasks x 3 splits (original / oneoff / conflicting) x 2 prompts (loose / strict).
Paper: arxiv.org/pdf/2609.19101. Reference for citing: numbers below are inline.

## Scorecard (one line)
- Logit-lens readout: REPLICATES (cleanest result).
- Detection (probe AUROC): REPLICATES the core claim.
- Probe ~= expensive monitor: DOES NOT replicate.
- Honeypot steering (causal): DOES NOT replicate.

## P(reward hack) -- OLMo is a "competent naive hacker"
- Tries to hack ~46% of impossible tasks under a loose prompt (counting enacted + considered, the paper's
  broad definition) -- roughly frontier-level propensity. It writes actual hacking code in ~38% of runs.
- But it only ever reaches for the crude hack: 100% of its hacks were hardcoding the tested inputs
  (`if input == X: return expected`). It never does the sophisticated harness/test-gaming that frontier
  models use.
- So it succeeds ~29% on oneoff (where hardcoding can actually pass) but only ~0-15% on conflicting
  (where you need `__eq__`/state tricks it doesn't know). Compared to the team's other models: like
  Qwen3.8-27B it basically can't pass conflicting; unlike Qwen (2% enacted) it enacts a lot (38%), closer
  to gpt-oss-120b's 19-42% -- but without gpt-oss's ability to actually pass conflicting (26%). OLMo wants
  to cheat like gpt-oss but can only pull it off like Qwen.
- Capability is fine: it solves 74-88% of normal (solvable) tasks and never hacks them (0% on controls --
  the judge is not trigger-happy). So the hacking isn't "it can't code."
- The naive pass-rate metric is misleading: on conflicting it reads ~0% because the task is unpassable
  even when hacked, yet the model is visibly hardcoding. You need the CoT judge to see the behavior.
- The anti-hack prompt matters a lot: it roughly halves enacted hacking (38% -> 21%) and kills oneoff
  success (29% -> 0%), converting much of it into explicit honest refusals.
- Verbatim example (conflicting, loose prompt): `if n == 1: return [0]  # Force to pass the conflicting test`.
  Models are also clearly eval-aware (they reason about "the test cases", "the grader").

## Logit-lens on the DoM vector -- REPLICATES (cleanest result)
- The difference-of-means reward-hacking vector decodes (via logit lens) to cheating words -- bypass, fake,
  cheat, dummy, pretend, artificially -- strongest around layers 28-40. None of those words appear in the
  synthetic data used to build it, so the direction learned the concept of cheating on its own.
- Same on Instruct (bypass, dummy, fake, mock, artificially). This is the paper's Table-1 result, on a 32B.

## Detection (probe AUROC) -- REPLICATES the core claim
- The vector detects hacking in real rollouts at AUROC 0.78 (held-out) / 0.84 (best config); chance is 0.5,
  and this is inside the paper's own range (0.74-0.97).
- A trained logistic-regression probe is about the same (0.83), so difference-of-means is enough.
- Built from 211 synthetic hack/honest minimal pairs (the paper's 5 families + hardcoding), SWE-agent-style,
  evaluated on LCB rollouts -- i.e. a cross-distribution transfer, which is itself the paper's claim.

## Probe vs monitor -- DOES NOT replicate
- The paper's selling point: the near-free probe matches an expensive LLM monitor. Ours doesn't -- at the
  monitor's low false-alarm rate (~10% FPR) the probe catches ~52% of hacks vs the monitor's ~81%.
- So the probe works, but it's clearly weaker than a strong Sonnet monitor. Likely cross-distribution
  transfer + 32B scale + small synthetic set (211 pairs vs the paper's ~5,285) + tail-truncated scoring of
  very long (40-60k token) rollouts.

## Honeypot steering (causal) -- DOES NOT replicate
- Setup: offer the model a cheat helper `zzz('solution.py')` that returns the correct answer, tell it NOT
  to use it, push it along the reward-hacking direction, and see if it starts calling the helper.
- On Think: unmeasurable -- it reasons for thousands of tokens and never reaches the code within a feasible
  generation budget, so there's nothing to measure.
- On Instruct (writes code directly): null. 0 of 48 take the bait at every steering strength (-8 to +8);
  negative-steer and random-direction controls are also ~0. (Paper: 0.5% -> ~25%.)
- It's a real null, not a bug: cranking the strength up visibly changes the output (coherent code at low
  strength, garbled at medium, pure gibberish by high), so amplifying the direction breaks the model's
  writing before it ever flips behavior. There's no "coherent-but-cheating" sweet spot.
- Could-be-our-fault caveats: our "don't use it" framing is stricter than the paper's (which mostly just
  announced the helper), we only tried one layer (L40) and a coarse strength grid, and n=48 rules out the
  paper's big effect but not a small one. So it's "no effect under our faithful-but-untuned setup," not
  "steering is impossible."

## Caveats / limitations (for the robustness reflection)
- One benchmark (Impossible-LCB); we did not run the SWE-V "natural" environment.
- Minimal scaffold only -- no test-file to edit, so the richest frontier hacks aren't even available; this
  is partly why OLMo only hardcodes.
- Small synthetic set (211 vs ~5,285); detection eval n=112; steering n=48/condition -- CIs are wide.
- Cross-distribution (synthetic SWE-agent pairs -> LCB rollouts) at small scale.
- Steering is a single-layer, untuned, strictly-framed test (see steering caveats above).

## Not done (future work / the proposal's extensions)
- A second big model end-to-end (GLM-4.5-Air was the plan; the team covered Qwen3.8-27B + gpt-oss-120b).
- Direction decomposition (is the vector spurious? regress onto eval-awareness / general-cheating /
  test-vocab sub-directions), and checkpoint tracking across OLMo's 47 RLVR checkpoints (the novel one --
  nobody has tracked a reward-hack direction across OLMo post-training), probe-penalty RL, J-lens.

## Where the files are (repo: github.com/lijuliana/2881r-mini-project)
- P(hack): results/exp1/FINAL_RESULTS.md (+ transcripts_full.jsonl, labels_consensus.jsonl).
- Probe (logit-lens + detection + probe-vs-monitor): results/probe/RESULTS.md.
- Instruct steering: results/probe/INSTRUCT_STEERING.md.
- Code: exp1/ (rollouts + judge), synth/ (pair generator), probe/ (extraction, probe analysis, steering).

## Numbers most likely to cite
- P(hack) OLMo Think, loose impossible: ~46% propensity, ~38% enacted, ~22% success (29% on oneoff).
- Capability: 74-88% control solve; 100% of hacks were hardcoding.
- Detection AUROC: 0.78 held-out / 0.84 best (paper 0.74-0.97).
- Probe vs monitor: 52% vs 81% TPR at ~10% FPR.
- Logit-lens tokens: bypass, fake, cheat, dummy, pretend, artificially (both models).
- Honeypot steering: 0/48 (null); output degrades to gibberish at high strength.
- Frontier reference (paper, judge-based, ImpossibleBench): Kimi 64.9% / GLM 50.0% / Qwen 64.8%.
