# Ahnaf's GSM8K Static-K Results: How AdaGRPO Should Use Them

## Purpose

These results provide historical motivation and diagnostic targets for AdaGRPO. They show that static K increases training cost, but accuracy does not improve steadily with K. They do not provide a directly matched baseline for the current AdaGRPO V2 checkpoints.

## 30-episode GSM8K study

The supplied figure PDF reports seed 42 and 30 training episodes. The accompanying project summary says the runs used Llama 3.2 3B, had 22 distinct training questions, 10 validation questions, and 30 held-out questions. The plots report:

| Static K | Training pipeline evaluations | Training time (min) | Best validation EM (%) | Held-out EM (%) | All-equal-reward groups (%) | Nonzero-gradient episodes (%) | Mean distinct pipelines/group | Unique/K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 150 | 11.14 | 70 | 76.67 | 53.33 | 46.67 | 3.67 | 0.73 |
| 8 | 240 | 15.91 | 50 | 66.67 | 50.00 | 50.00 | 4.73 | 0.59 |
| 10 | 300 | 15.91 | 60 | 76.67 | 40.00 | 60.00 | 5.47 | 0.55 |
| 15 | 450 | 27.61 | 50 | 70.00 | 43.33 | 56.67 | 5.63 | 0.38 |

The validation set had only ten questions, so each validation answer changed accuracy by ten points. K=5 and K=10 tied on the 30-question held-out set despite twice as many training pipeline evaluations for K=10. K=15 produced more unique pipelines per group in absolute terms, but its unique-pipeline fraction was lower.

## 200-episode GSM8K study

The supplied source metadata identifies seed 42, 200 episodes, 200 training questions, 50 validation questions, Llama 3.1 8B, learning rate 1e-5, and the same pinned GSM8K dataset revision used by the current research code. It also records semantic fallback enabled, so its reward protocol does not match the current numeric-only V2 protocol.

| Static K | Training evaluations | Training time (min) | Logical LM calls | Zero-advantage groups (%) | Nonzero-gradient episodes (%) | Mean distinct pipelines/group | Unique/K | Fixed episode-200 test EM (%) | Validation-selected test EM (%) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1,000 | 66.10 | 1,119 | 87.5 | 11.5 | 2.02 | 0.40 | 78.5 | 78.5 |
| 8 | 1,600 | 106.60 | 1,853 | 84.5 | 14.5 | 2.32 | 0.29 | 74.5 | 79.0 |
| 10 | 2,000 | 129.17 | 2,352 | 85.0 | 13.5 | 2.35 | 0.24 | 72.5 | 72.5 |
| 15 | 3,000 | 189.90 | 3,390 | 89.5 | 10.0 | 2.00 | 0.13 | 82.0 | 82.0 |

The fixed episode-200 checkpoint scores are not always the same as scores from the validation-selected checkpoints. The charts label the latter checkpoint episodes as 50, 50, 200, and 150 for K=5/8/10/15. Use validation selection for checkpoint choice, then reserve final testing for the selected checkpoint.

There is a test-size discrepancy to resolve before citing the 200-episode held-out results: the supplied sources metadata says `test_size=200`, and the plotted accuracy increments are 0.5 percentage points, while the accompanying written summary says 50 test questions. The raw evaluation manifest was not included with the supplied figure bundle. Do not repeat the 50-question claim as verified without checking the original run artifact.

## What this contributes to AdaGRPO

1. **Motivation for adaptive allocation:** K=15 used three times as many training pipeline evaluations as K=5 in the 200-episode study and took about 2.9 times as long. Larger K can buy some accuracy in these runs, but the result is inconsistent across K and checkpoint choice.
2. **A reason to inspect reward information:** 84.5–89.5% of the 200-episode groups had equal rewards, and only 10–14.5% had a recorded nonzero gradient. More samples alone did not remove this issue. AdaGRPO should measure whether additional candidates produce useful class-level reward contrast and report actual gradient behavior.
3. **Keep a same-budget static control:** Static-K=4 is essential for distinguishing adaptive allocation from simply spending fewer attempts. The current matched V2 study includes it alongside Static-K=5.
4. **Compare costs in several units:** Report pipeline evaluations, logical LM calls, time, and tokens when available. The old plots show that pipeline count and model-call count are related but not identical.
5. **Keep studies protocol-matched:** Do not pool these scores with current V2 results. The historical 200-episode runs used Llama 3.1 8B, 200 training episodes/questions, and semantic fallback. Current V2 uses Llama 3.2 3B, a 12-episode development study, 50 training questions, and numeric exact-match rewards. The 30-episode run also has a much smaller training/validation/test sample.

## Completed checkpoint evaluation (fresh 100)

The inference-only evaluation compared validation-selected seed-43 V2 Adaptive-K, Static-K=4, and Static-K=5 checkpoints on the same 100 GSM8K training-split questions. The corrected membership excludes the reconstructed seed-42 200-training/50-validation set and all recorded or reserved historical memberships. It performed no training and did not load the official test split. The protocol and per-question artifacts are in `results/adagrpo_fresh100/corrected_v2/`; the question membership SHA-256 is `a05933abad79ff2cffc8562e6b1e954dc73e7f2a6a03c1e711c1c295d40e80ed`.

| Checkpoint | Selected episode | Correct / 100 | Accuracy | Logical model calls | Evaluation attempts | Evaluation execution seconds | Training attempts already spent to produce source checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive V2 | 12 | 68 | 68% | 105 | 100 scored; 101 gross starts | 1,386.6 | 42 |
| Static-K=4 | 0 | 66 | 66% | 114 | 100 | 1,539.9 | 48 |
| Static-K=5 | 0 | 66 | 66% | 114 | 100 | 1,537.7 | 60 |

On the paired questions, Adaptive V2 scored two percentage points above each static checkpoint: 13 wins, 76 ties, and 11 losses against either arm. The 10,000-resample paired question bootstrap intervals were −8 to +11 percentage points against Static-K=4 and −7 to +11 points against Static-K=5. These wide intervals do not establish a reliable advantage.

There is a key checkpoint-selection limitation: Static-K=4 and Static-K=5 both selected episode 0, and their policy digests are identical. Thus those two rows are the same initial policy checkpoint evaluated twice, not two learned static policies. This run answers how the saved V2 episode-12 policy compares with that initial policy on these questions. It does not establish that AdaGRPO is better than trained Static-K baselines.

Adaptive incurred one interrupted/repeated evaluation attempt (a `KeyboardInterrupt` during the earlier save-error recovery); it is included in the 101 gross attempts. All three final arms contain 100 unique scored answers. The adaptive arm reported 105 logical calls; each static arm reported 114. Physical HTTP request counts were unavailable. Numeric extraction used the last-number fallback for 57 adaptive answers and 52 answers in each static arm; four adaptive and seven static outputs per arm were marked unparseable by the exact parser. This is evidence to inspect the parser and model output format, not a reason to change scoring after seeing results.

Use this evaluation as a fresh-question generalization diagnostic for the saved adaptive checkpoint and as a reproducibility check. Do not merge its accuracy with Ahnaf's 30- or 200-episode tables, claim formal test performance, or count inference attempts as training-compute savings. The historical tables remain useful for motivating adaptive allocation, identifying diminishing returns and reward-signal sparsity, and defining matched future controls. The next fair performance comparison needs trained static checkpoints under the same numeric reward, model, training schedule, and selection protocol; the current evaluation does not replace that experiment.

## Supplied sources

- `C:/Users/rahma/Downloads/GSM8K_static-k-comparison_figures.pdf` — 30-episode figure collection.
- `C:/Users/rahma/Downloads/GSM8K_static-k-comparison_figures/episode_200/all_in_one_figures.pdf` — 200-episode figure collection.
- `C:/Users/rahma/Downloads/GSM8K_static-k-comparison_figures/best_checkpoint/all_in_one_figures.pdf` — validation-selected 200-episode checkpoint results.
- `C:/Users/rahma/Downloads/GSM8K_static-k-comparison_figures/sources.json` — recorded run and model configuration for the 200-episode collection.
