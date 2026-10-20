# AdaGRPO handoff — state as of 20 October 2026

This document exists so any fresh agent session can continue without re-deriving or inventing anything. Read it before touching anything. All numbers below are copied from committed run artifacts, not from memory.

## 1. Authorization chain

- The user approved `PLAN.md` (root, also mirrored in Downloads) on 20 October 2026: implement **AdaGRPO** — automatic bounded per-question K selection for AutoDSPy's GRPO on GSM8K — on branch `amanur`, preserving every original notebook and historical result.
- `docs/DECISIONS.md` records: **D029** (implementation authorization + full protocol) and **D030** (pilot wall-limit handling + user-approved resume deviation + screening outcome).
- Hard rule: the research hypothesis is **≥20% fewer gross training attempts than matched Static-K=5 with accuracy within 2 points**, provable only via the formal study (seeds 42/43/44, arms adaptive/static2/static4/static5, full official test split, paired hierarchical bootstrap). Nothing so far proves or disproves it.

## 2. What was built (coding is COMPLETE)

- `AdaGRPO_GSM8K.ipynb` — self-contained. Adaptive allocation + matched static K=2/4/5 controls in one implementation. Preserves the original Figure-2 summed loss, empirical group-centered advantages, pretrained GPT-2 policy (revision 607a30d…), module/signature/stop grammar with first-token collisions, one scheduled question per episode, sampling with replacement, sequential differentiable replay, LR 1e-5, Adam `foreach=False`. No clipping/entropy/normalization/KL added.
- **Controller** (`controller_decision`): probe 2 → expand 2, bounds 2..15, stop rules `useful_contrast` / `all_correct` (≥4) / `policy_concentrated` (≥4 and Neff ≤ 1.25) / `allowance_reached`; effective-class credit contrast across colliding module/signature paths (threshold 1e-12); dynamic Neff from exact joint probabilities without RNG consumption; reserve 2 attempts per remaining episode before expanding the current one.
- **Durability**: pre-execution journaling (`attempts.jsonl` starts/finishes), atomic best/latest checkpoints with SHA-256 pointers, truncated-journal-tail repair, resume with budget reconciliation from the journal, wall-limit handling that discards (but charges) the partial group.
- **Accounting**: actual-K everywhere (summaries, validation provenance, plots, audits); gross costs include failed and discarded attempts; runtime identity (packages, model digest, hardware) recorded per run.
- **Ops**: `run_local_suite("smoke"|"pilot")`, `run_operation("initialize"|"train"|"resume"|"test"|"compare")`, `run_cloud_smoke(root, allow_cloud_training=False)`, `run_formal_study(..., allow_formal_training=True)` — the last two raise unless explicitly opted in. `test_model` requires completed formal training. Definition cells launch nothing.
- Literature mapping (idea → paper): probe/expand ← AERO (2602.14338); explicit bounded stops ← Reinforce-Ada (2510.04996); progressive budget/reservation ← VIGOR (2607.22002); measure-real-gradient diagnostics ← GVM (2505.02391); sampling-bias stance + by-refusal design ← DPPO (2603.04135). DAPO/SAGC/GRPO-LEAD excluded on purpose. The loss/advantages/update were never touched by any paper. See `docs/ADAPTIVE_K_PAPER_GUIDE.md`.

## 3. Environment (recreate before any run)

- Python: `.\venv_adagrpo\Scripts\python.exe` (3.11.9, torch 2.6.0+cu124, transformers 5.17.0, dspy 3.3.1, datasets 5.0.1). Kernel name `AdaGRPO GPU`.
- Execution model: `llama3.2:3b` via Ollama (host http://localhost:11434, digest a80c4f17…), CPU offload (`ollama_num_gpu=0`), ctx 2048, max output 512, temperature 0. Policy on CUDA FP32. Hardware: RTX 3050 Laptop 6GB, 16GB RAM, i5-13420H. Never substitute models. Full freeze: `docs/audit_artifacts/adagrpo_environment.txt`.

## 4. Verification state (all regenerated against current notebook sha256 bcbb65ec…)

- `docs/audit_artifacts/adagrpo_self_checks.json` — **93/93 synthetic checks pass** (no Llama, no downloads): loss/gradient/Adam equivalence, collision cancellation, controller rules, reservation, truncated-tail repair, recovery, tamper detection, formal opt-in guards, bootstrap cases.
- `docs/audit_artifacts/adagrpo_clean_kernel.json` — all 19 code cells execute with zero operations launched.
- `docs/audit_artifacts/adagrpo_preservation.json` — all **121** original notebooks/results files hash-identical. Never modify anything outside the AdaGRPO namespace.
- `docs/audit_artifacts/adagrpo_artifact_audits.json` — `verify_run_artifacts` **passes on all five live runs** (both smokes, static pilot, completed adaptive pilot incl. resumed segment, recovery run).
- `docs/audit_artifacts/adagrpo_live_recovery.json` — real CUDA interruption-after-durable-episode-1 + resume: Adam steps exactly {2}, 8/8 attempts, finite gradients, no test access; independently re-verified from a fresh process.

## 5. Live runs and results (all under `results/adagrpo/`)

- **Smoke** (2 episodes/method, real models): static 10 attempts, adaptive 8; both integrated correctly. `adagrpo_live_smoke.json`.
- **Pilot** (seed 42, 20 episodes, 50 train / 20 validation questions, shared 60-min suite clock):
  - `static_k5/seed42/pilot/20261020T105201-bb79e00d` — **completed**: 100/100 attempts, best validation **0.95 @ ep10** (0.80→0.95→0.85), 13/20 zero-advantage groups, 7/20 informative gradients, ~3.08 GB policy VRAM, zero parse/candidate/journal failures.
  - `adaptive/seed42/pilot/20261020T113252-73f1340a` — stopped by the shared clock at 10/20 (36/80 attempts), then **resumed from the durable ep-10 checkpoint with explicit user approval** (D030) under a fresh per-segment wall limit. **Completed**: 80/80 attempts, zero discarded, stops 11 useful-contrast / 5 all-correct / 4 allowance-reached, K used {2,4,5,15}, 9/20 zero-advantage, 11/20 informative gradients, ~3.10 GB, zero failures.
- **Screening verdict: NOT PROMISING as-is.** Compute met the target exactly (80 vs 100 = 20% fewer). Accuracy failed: adaptive best stayed **0.80 @ ep0** (all checkpoints 0.80/0.80/0.80) vs static 0.95 — a 15-point gap vs the 5-point allowance.
- **Mechanisms identified** (infrastructure was clean): (1) policy concentration — neff fell 3–8 → 2.1–2.7 in the resumed segment; (2) budget drain — episode 11 spent the full 15-attempt allowance on an all-failure group with zero gradient (18.75% of the run budget on no credit), after which reservation math forced K=2 groups at eps 17–19; (3) a lower zero-advantage rate (9 vs 13) bought nothing — the plan's standing caution.
- Details: `docs/audit_artifacts/adagrpo_pilot_diagnostics.json`, `adagrpo_adaptive_resume.json`, suite record `results/adagrpo/pilot_suite_b7c48a6c.json`.

## 6. Deviations and known anomalies (do not hide these)

- The pilot's 60-min limit was implemented as a **shared suite clock**; the adaptive arm paid for static's 40.6 min. Recorded in D030; future suites should budget per arm.
- The resume was a **user-approved deviation** (D030): a dependent second wall segment, labeled resumed, not a single-shot bounded pilot.
- Both long-running launcher processes exited code 1 **after** printing success, with no traceback. Every affected artifact was re-verified from fresh processes; the anomaly is noted inside the audit JSONs. If it recurs, verify artifacts rather than re-running blindly.
- The wall-limited summary.json was overwritten by the completed resume (by design); the partial-run history lives in the journal and the interim audit artifacts.

## 7. Current repo state

- Branch `amanur`. **Everything is uncommitted**: `AdaGRPO_GSM8K.ipynb`, 4 research/report docs, `docs/audit_artifacts/adagrpo_*.json` (7 files), `results/adagrpo/**` (incl. checkpoints — `.pt` is gitignored; copy whole run dirs to preserve them), plus expected-dirty `graphify-out/` (dirty graph files are normal; not a reason to skip graphify queries).
- No final-test access ever occurred (`test_count=0` everywhere; no `test_*.json` in any AdaGRPO run dir).

## 8. What remains — none of it authorized to run automatically

1. **Static-K=4 matched control** (same 80-attempt budget, ~30–40 min local): the decisive "budget cap vs adaptivity" question before any controller redesign.
2. **Controller redesign candidates** (require explicit user approval + matched reruns): no-progress allowance cap / circuit-breaker for zero-credit groups; concentration countermeasures.
3. **Colab formal study** (D029 spec; ~27,828 pipeline executions; explicit opt-in only) — currently blocked behind resolving the pilot failure.
4. Commit decision belongs to the user.

## 9. Rules for any future agent

- Read `AGENTS.md`, `docs/agents/*` before ticket/triage/domain work. Run `graphify query` for codebase questions.
- Never invent or fabricate results; report from artifacts only. Never launch training without explicit user approval. Never touch historical notebooks/results. Never substitute models. Verify GRPO math against `assets/Figure_2 (F).-1.png` semantics when in doubt.
- If something looks missing, check `docs/audit_artifacts/` first — it is the source of truth for what was measured.
