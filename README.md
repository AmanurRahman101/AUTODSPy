# AdaGRPO

### Automatic candidate allocation for AutoDSPy

AdaGRPO learns a DSPy pipeline policy with GPT-2 and automatically chooses how many candidate pipelines to evaluate for each GSM8K training question. This repository contains the original AutoDSPy reference, Static-K benchmarks, adaptive controllers, and their experimental evidence.

**Status - 21 October 2026:** implementation and local experiments are complete. Accuracy-preserving compute savings have **not** been established. V2 remains the main adaptive reference; the latest V3 screen did not justify a larger experiment.

## Start here

- **Understand the findings:** [current research status](docs/research/STATUS.md).
- **Find the right notebook:** [notebook catalog](docs/catalogs/NOTEBOOKS.md).
- **Find a result or its evidence:** [results catalog](docs/catalogs/RESULTS.md).
- **Set up the laptop:** [local environment guide](docs/guides/LOCAL_SETUP.md).
- **Find everything else:** [documentation index](docs/README.md) and [project map](docs/catalogs/PROJECT_MAP.md).
- **Read the literature:** [paper library](docs/research/PAPERS.md).

## Research target

Demonstrate at least **20% fewer gross training pipeline attempts than matched Static-K=5**, with no more than a **two percentage point accuracy decrease**, supported by the predefined paired uncertainty analysis.

Automatic K selection is implemented. Demonstrating the accuracy/cost target remains experimental work. Outcomes from different seeds, reward protocols, models, or question memberships must be interpreted separately.

## Main notebooks

- [Frozen V2](AdaGRPO_GSM8K_v2.ipynb): bounded adaptive K with candidate-specific random streams.
- [Verified V2](AdaGRPO_GSM8K_v2_verified.ipynb): study recovery and accounting around the frozen V2 method.
- [V3 experiment](AdaGRPO_GSM8K_v3.ipynb): five initial K=5 groups followed by V2 allocation; completed screen failed its advancement gate.
- [Original GRPO reference](DSPy_GRPO.ipynb): preserved upstream implementation.
- [Static-K GSM8K](Static_K_GRPO_GSM8K.ipynb) and [HotpotQA](Static_K_GRPO_HotpotQA.ipynb): historical baseline notebooks.

Every notebook remains self-contained. Definitions in the newer AdaGRPO notebooks do not launch training automatically. Open notebooks from the repository root using the existing **AdaGRPO GPU** kernel. Original reference notebooks have their own execution behavior.

## Latest completed screen

V2 and V3 each trained for 30 episodes on a shared seed-42 schedule. V2 used **92 training attempts**; V3 used **97**. Both selected the episode-0 checkpoint and scored **12/20** on confirmation questions. Validation fell from **7/10 to 4/10** after training in both arms.

This establishes no benefit from V3's warm-up in this screen. Matching initialization does not demonstrate useful policy learning. See the [audited closeout](results/adagrpo_v3/closeout_audit.json) and [interpretation](docs/research/STATUS.md).

## Repository layout

```text
AUTODSPy/
|-- README.md                    Project entry point
|-- *.ipynb                      Stable, self-contained notebook entry points
|-- AutoDSPy.pdf                 Foundational paper
|-- docs/
|   |-- README.md                Documentation index
|   |-- catalogs/                Notebook, result, and file navigation
|   |-- guides/                  Local setup and operation boundaries
|   |-- research/                Current status and paper library
|   |-- reference/               Preserved documentation before cleanup
|   |-- audit_artifacts/         Verification evidence
|   |-- agents/                  Agent workflow instructions
|   `-- *.md                     Existing reports and decisions at stable paths
|-- results/                     Separate experiment namespaces
|-- assets/                      Original paper figures
|-- graphify-out/                Generated knowledge graph
|-- venv_adagrpo/                Local research environment; excluded from Git
`-- venv/                        Earlier local environment; excluded from Git
```

The larger literature library is in the sibling folder **`../papers/`**, outside this Git repository. Notebook and result paths stay stable because frozen configurations, recovery code, and historical manifests refer to them.

## Method and evidence

AdaGRPO preserves the original summed group-centered loss, numeric binary rewards in the new protocol, GPT-2 first-token action scoring, learning rate `1e-5`, and existing Adam behavior. Outcome-dependent stopping changes the sampling distribution; an unchanged loss expression does not establish fixed-K estimator equivalence.

Use [DECISIONS.md](docs/DECISIONS.md) for protocol changes and authorizations. Use each study's journal, ledger, and audit for costs and completeness. Earlier reports describe their own experiments; the current status page provides the cross-study interpretation.

## Checkpoints and sharing

Model weights (`*.pt`), environments, and some local artifacts are excluded from Git. Sharing the repository alone does not transfer a resumable model run. Copy the **complete run directory**, including weights, journals, manifests, and optimizer checkpoints. See [results handling](results/README.md).

## Original research

AutoDSPy was published in the **EMNLP 2025 Industry Track**. The official record and PDF are available from [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.192/). Corrected citation metadata is in [CITATION.cff](CITATION.cff).

[Original README before organization](docs/reference/README_BEFORE_ORGANIZATION.md) is preserved for provenance; it contains historical instructions and metadata that should not override current experiment guides.
