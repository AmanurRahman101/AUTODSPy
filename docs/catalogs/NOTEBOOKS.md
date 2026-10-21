# Notebook catalog

Notebook filenames and locations are stable interfaces. Run with the repository root as the working directory. This catalog groups their roles without moving experimental sources or changing frozen code hashes.

## Adaptive controller implementations

- [AdaGRPO_GSM8K.ipynb](../../AdaGRPO_GSM8K.ipynb) - original V1 controller and local engineering foundation; historical reference.
- [AdaGRPO_GSM8K_v2.ipynb](../../AdaGRPO_GSM8K_v2.ipynb) - frozen V2 bounded refinement controller. Main adaptive methodological reference.
- [AdaGRPO_GSM8K_v2_verified.ipynb](../../AdaGRPO_GSM8K_v2_verified.ipynb) - frozen V2 plus stronger study orchestration, recovery, accounting, and formal-study preparation. Infrastructure improvements do not imply higher accuracy.
- [AdaGRPO_GSM8K_v3.ipynb](../../AdaGRPO_GSM8K_v3.ipynb) - completed early-exploration experiment with a frozen V2 control. Its local screen failed the advancement gate; retain as evidence rather than promote it as an improved model.

## Comparisons and checkpoint evaluation

- [AdaGRPO_V1_V2_Comparison.ipynb](../../AdaGRPO_V1_V2_Comparison.ipynb) - matched controller comparison across seeds 43/44; observed V2 advantage remained statistically inconclusive.
- [AdaGRPO_GSM8K_Fresh100_v2.ipynb](../../AdaGRPO_GSM8K_Fresh100_v2.ipynb) - corrected 100-question evaluation of existing V2/static checkpoints, without retraining.
- [AdaGRPO_GSM8K_Fresh100.ipynb](../../AdaGRPO_GSM8K_Fresh100.ipynb) - superseded first attempt after membership overlap was found. Preserve for cost accounting; use the corrected notebook/report for findings.
- [AdaGRPO_GSM8K_Historical30.ipynb](../../AdaGRPO_GSM8K_Historical30.ipynb) - retrospective comparison with the archived 30-episode Static-K baseline; reward/sampling/runtime differences limit causal interpretation.
- [AdaGRPO_GSM8K_Matched30_Diagnosis.ipynb](../../AdaGRPO_GSM8K_Matched30_Diagnosis.ipynb) - new matched Static-K=5 control and initialization diagnosis for the 30-episode V2 run; uses previously examined evaluation questions.

## Historical Static-K baselines

- [Static_K_GRPO_GSM8K.ipynb](../../Static_K_GRPO_GSM8K.ipynb) - GSM8K fixed-K training and benchmark analysis.
- [Static_K_GRPO_HotpotQA.ipynb](../../Static_K_GRPO_HotpotQA.ipynb) - HotpotQA fixed-K experiment foundation; AdaGRPO V1-V3 studies focus on GSM8K.

## Original AutoDSPy references

- [DSPy_GRPO.ipynb](../../DSPy_GRPO.ipynb) - preserved original GRPO implementation.
- [DSPy_PPO.ipynb](../../DSPy_PPO.ipynb) - original PPO alternative.
- [DSPy_Reinforce.ipynb](../../DSPy_Reinforce.ipynb) - original REINFORCE alternative.

## Diagnostic notebook

- [docs/diagnostics/k5-20261013/Diagnosis.ipynb](../diagnostics/k5-20261013/Diagnosis.ipynb) - earlier Static-K=5 investigation.

## Choosing an entry point

To understand the current method, read V2. To inspect the latest completed experiment, read V3 and its [closeout](../../results/adagrpo_v3/closeout_audit.json). To inspect results without executing notebook cells, use the [results catalog](RESULTS.md).

Original notebooks and older experiments differ in initialization and launch behavior. Do not use Run All as a read-only inspection operation. The frozen V3 screen is finished and failed its gate; new model work requires a new approved protocol.
