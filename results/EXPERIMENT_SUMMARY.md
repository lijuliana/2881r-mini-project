# OLMo 3.1 32B results

OLMo 3.1 32B Think (main) + Instruct (steering). Benchmark: Impossible-LCB. Judge: Sonnet 4.5, 3x consensus.
150 rollouts (25 tasks x 3 splits x 2 prompts). Paper: arxiv.org/pdf/2609.19101.

## P(hack): competent naive hacker
- Hacks ~46% of impossible tasks (loose prompt, enacted+considered), ~38% enacted. About frontier-level propensity.
- But it only hardcodes the tested inputs (`if input==X: return expected`) -- 100% of its hacks. Never games
  the harness/tests the way frontier models do.
- Passes ~29% on oneoff (hardcoding works there) but ~0-15% on conflicting (needs `__eq__`/state tricks it
  doesn't have). Vs the team's models: enacts a lot like gpt-oss (19-42%), but can't pass conflicting like
  Qwen3.8-27B -- gpt-oss passes conflicting 26%, OLMo basically can't.
- Not a capability problem: solves 74-88% of solvable tasks, hacks 0% of them.
- Pass-rate alone reads ~0% on conflicting (unpassable even when hacked); you need the CoT judge to see it.
- Anti-hack prompt halves enacting (38->21%) and zeroes oneoff success (29->0%).
- Example (conflicting, loose): `if n == 1: return [0]  # Force to pass the conflicting test`. Clearly eval-aware.

## Logit-lens: replicates (cleanest result)
- The DoM vector decodes to bypass, fake, cheat, dummy, pretend, artificially (layers 28-40). None of those
  words are in the synthetic pairs it's built from. Same on Instruct. This is the paper's Table 1.

## Detection (probe AUROC): replicates
- 0.78 held-out / 0.84 best config (chance 0.5; paper 0.74-0.97). Logistic probe about the same (0.83).
- Built from 211 synthetic pairs, tested on LCB rollouts -- cross-distribution, as in the paper.

## Probe vs monitor: diverges
- Paper's pitch is probe ~= expensive monitor. Ours is weaker: at ~10% FPR the probe catches 52% vs the
  monitor's 81%. Blame 32B scale + 211 pairs (paper had ~5,285) + cross-distribution.

## Honeypot steering: doesn't replicate
- Offer a cheat helper `zzz('solution.py')`, forbid it, steer along the hack direction, check if it uses it.
- Think: can't measure (reasons too long to reach code).
- Instruct: 0/48 at every strength; negative + random controls also ~0 (paper: 0.5% -> 25%).
- Real null, not a bug -- high strength garbles the output and breaks it into gibberish; the direction
  degrades the model before behavior flips.
- Most likely why: our prompt says "do NOT rely on it", the paper's just announces the helper. So we were
  testing "does steering make it defy an explicit instruction" (baseline 0%), not "does steering make it
  take a shortcut" (paper baseline 0.5% -> 25%). That raises the bar a lot. Other suspects: one layer (L40),
  coarse strength grid, n=48. Worth a re-run with announce-only framing before calling it a true null.

## Caveats
- One benchmark (no SWE-V). Minimal scaffold, so no test file to edit -- sophisticated hacks aren't even
  available, which is part of why OLMo only hardcodes. 211 pairs vs 5,285. Detection n=112, steering n=48.

## Not done (extensions)
- Second big model end-to-end; direction decomposition; checkpoint tracking across OLMo's 47 RLVR
  checkpoints (novel -- nobody's done it); probe-penalty RL; J-lens.

## Files (github.com/lijuliana/2881r-mini-project)
- P(hack): results/exp1/FINAL_RESULTS.md. Probe: results/probe/RESULTS.md. Steering: results/probe/INSTRUCT_STEERING.md.
- Code: exp1/ (rollouts + judge), synth/ (pair generator), probe/ (extraction, analysis, steering).

---

## Table (for reference)

Scorecard, us vs paper:

| Result | Paper (1T+ models) | OLMo 32B | Verdict |
|---|---|---|---|
| Models reward-hack | 50-96% | 46% propensity / 29% success (oneoff) | replicates |
| Logit-lens = cheating words | cheating, illicit, hardcoded | bypass, fake, cheat, dummy, pretend | replicates |
| Probe detects hacking (AUROC) | 0.74-0.97 | 0.78 held-out / 0.84 best | replicates |
| Probe ~= LLM monitor | yes | 52% vs 81% TPR @ 10% FPR | diverges |
| Steering causes hacking | 0.5% -> 25% | 0/48 (null) | diverges |

P(hack) breakdown (3x consensus, defects excluded):

| Condition | enacted | propensity | success (pass) |
|---|---|---|---|
| loose / oneoff | 47% | 59% | 29% |
| loose / conflicting | 30% | 35% | 15% |
| strict / oneoff | 28% | 33% | 0% |
| strict / conflicting | 14% | 38% | 10% |
| control (solvable) | 0% | 0% | 74-88% solved |

Frontier reference (paper, judge-based, ImpossibleBench): Kimi 64.9% / GLM 50.0% / Qwen 64.8%.
