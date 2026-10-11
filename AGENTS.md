## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).



# AutoDSPy Research — Agent Instructions

## Project Objective
Research static versus adaptive group-size selection
in AutoDSPy's GRPO algorithm.

Current scope: Static-K baseline experiments.

## Research References
- Paper: AutoDSPy.pdf
- Reference implementation: DSPy_GRPO.ipynb

Read the original notebook and relevant paper sections
before changing experimental methodology.

## Architecture Rules
- Preserve DSPy_GRPO.ipynb unchanged.
- Use self-contained Jupyter notebooks.
- Keep initialization, training, evaluation, and
  visualization inside each notebook.
- Do not introduce separate Python modules unless
  explicitly requested.
- Follow the original notebook's conceptual structure.

## Experimental Rules
- Policy network: GPT-2 127M.(can change from codebase for more experiments)
- Execution model: Llama 3.1 8B via Ollama.(can change from codebase for more experiments)
- Datasets: GSM8K and HotpotQA.
- Static K values: 5, 8, 10, 15.
- Training pool: 200 examples.(can change from codebase for more experiments)
- Training episodes: 200.(can change from codebase for more experiments)
- One randomly selected prompt per episode.
- Sample with replacement.
- Use the same saved episode schedule across K.
- Initialize each K experiment independently.

## GRPO Rules
- Preserve the GRPO loss defined in assests/Figure_2 (F).-1.png
- Use group-centered advantages: A_i = r_i - mean(r).
- Do not add PPO clipping, entropy regularization,
  GAE, KL penalties, or reward normalization.
- Document any change to the original methodology.

## Reproducibility
- Use fixed random seeds.
- Prevent training/validation/test leakage.
- Store configurations and experiment metadata.
- Save checkpoints independently for each K.
- Make interrupted training resumable.
- Record accuracy, rewards, loss, gradient norms,
  latency, pipeline evaluations, and GPU memory.
- Never invent or fabricate experimental results.

## Hardware
- Target GPU: RTX 3060 Laptop, 6GB VRAM.
- Avoid unnecessary GPU model duplication.
- Execute candidate pipelines sequentially.
- Support Ollama CPU offloading when required.
- Never silently substitute a different LLM.

## Development Workflow
- Work incrementally.
- Preserve existing work and uncommitted changes.
- Verify GRPO mathematics and gradient propagation.
- Run lightweight tests before expensive experiments.
- Never launch full training automatically.
- Report completed changes, verification results,
  deviations, and unresolved issues.

## Agent skills

### Issue tracker

Track issues in GitHub Issues for AmanurRahman101/AUTODSPy.
Before ticket operations, read `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels.
Before triage, read `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root `GLOSSARY.md` and `docs/adr/`.
Before domain exploration, read `docs/agents/domain.md`.
