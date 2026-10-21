# Paper library

## Location and scope

Nine PDF files are present in the sibling **`CSE498R/papers/`** folder: AutoDSPy plus eight related papers. The sibling folder is outside this repository and will not accompany a Git clone automatically. Links below target the local copies; use the [existing paper guide](../ADAPTIVE_K_PAPER_GUIDE.md) for public download links and detailed methodological cautions.

This catalogs the supplied local library. Descriptions summarize the papers' documented role in our project; they are not fresh publication-status checks or claims that those methods were fully implemented here.

## Foundation and related work

- **[AutoDSPy](../../../papers/AutoDSPy_EMNLP2025.pdf)** - foundational framework and original GRPO objective.
- **[AERO](../../../papers/AERO_2602.14338.pdf)** - probe and bounded expansion inspiration.
- **[Reinforce-Ada](../../../papers/Reinforce-Ada_2510.04996.pdf)** - explicit adaptive sampling/stopping inspiration.
- **[VIGOR](../../../papers/VIGOR_2607.22002.pdf)** - budget allocation inspiration.
- **[GVM](../../../papers/GVM_2505.02391.pdf)** - gradient-related diagnostics.
- **[DPPO](../../../papers/DPPO_2603.04135.pdf)** - sampling-bias and estimator analysis.
- **[DAPO](../../../papers/DAPO_2503.14476.pdf)** - supporting comparison; filtering not imported.
- **[SAGC](../../../papers/SAGC_2606.02218.pdf)** - supporting comparison; parallel control not imported.
- **[GRPO-LEAD](../../../papers/GRPO-LEAD_EMNLP2025_2025.emnlp-main.287.pdf)** - supporting comparison; reward changes not imported.

A second foundational copy is kept as [AutoDSPy.pdf](../../AutoDSPy.pdf) at the repository root. The official publication record is [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.192/); it is the EMNLP 2025 Industry Track paper, DOI `10.18653/v1/2025.emnlp-industry.192`.

## How to use this library

Start with AutoDSPy for the original system, then AERO and Reinforce-Ada for allocation ideas. Read VIGOR, GVM, and DPPO for budget/diagnostic/bias considerations. The existing [research understanding](../ADAPTIVE_K_RESEARCH_UNDERSTANDING.md) and [decisions record](../DECISIONS.md) explain which ideas were adopted.

The implemented controllers selectively borrow ideas. They do not inherit another paper's estimator guarantees. DAPO prompt filtering, SAGC parallel control, and GRPO-LEAD reward changes were outside the frozen V1-V3 controller experiments.
