# Graph Report - AUTODSPy  (2026-10-10)

## Corpus Check
- 30 files · ~55,483 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 4, .ipynb 4, .cff 1)

## Summary
- 309 nodes · 534 edges · 21 communities (16 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `50e572d7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- reward.py
- runtime.py
- Findings and proposed corrections
- DSPy Pipeline Optimization with Reinforcement Learning
- What You Must Do When Invoked
- test_reproduction.py
- Execution
- Policy
- 📊 CSE 498R Thesis: Baseline Execution & Empirical Analysis Report
- run_grpo_baseline.py
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- AGENTS.md
- extraction-spec.md
- __main__.py

## God Nodes (most connected - your core abstractions)
1. `train()` - 17 edges
2. `evaluate()` - 13 edges
3. `compute_reward()` - 13 edges
4. `main()` - 12 edges
5. `Config` - 12 edges
6. `seed_everything()` - 12 edges
7. `What You Must Do When Invoked` - 12 edges
8. `Findings and proposed corrections` - 11 edges
9. `DSPyExecutor` - 10 edges
10. `Policy` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Model and dataset setup` --references--> `train()`  [INFERRED]
  docs/REPRODUCTION_GUIDE.md → autodspy_reproduction/training.py
- `RunTests` --uses--> `DSPyExecutor`  [INFERRED]
  tests/test_reproduction.py → autodspy_reproduction/execution.py
- `PolicyTests` --uses--> `Config`  [INFERRED]
  tests/test_reproduction.py → autodspy_reproduction/config.py
- `RunTests` --uses--> `Config`  [INFERRED]
  tests/test_reproduction.py → autodspy_reproduction/config.py
- `RunTests` --uses--> `Example`  [INFERRED]
  tests/test_reproduction.py → autodspy_reproduction/data.py

## Import Cycles
- None detected.

## Communities (21 total, 5 thin omitted)

### Community 0 - "reward.py"
Cohesion: 0.14
Nodes (6): canonical_answer(), exact_match(), normalize_number(), parse_judge_score(), Reward, NumericalTests

### Community 1 - "runtime.py"
Cohesion: 0.18
Nodes (11): evaluate(), compute_reward(), append_json(), artifact_hash(), cuda_memory(), json_write(), load_checkpoint(), new_run() (+3 more)

### Community 2 - "Findings and proposed corrections"
Cohesion: 0.06
Nodes (29): Deviations, assumptions and unresolved evidence, Hardware-only implementation choices, Methodological assumptions and repairs, Readiness criterion, Sensitivity modes, not reproduction defaults, Verified choices, Actions and pipeline termination, Checkpointing and reproducibility (+21 more)

### Community 3 - "DSPy Pipeline Optimization with Reinforcement Learning"
Cohesion: 0.07
Nodes (28): 1. PPO (Proximal Policy Optimization), 2. GRPO (Group Relative Policy Optimization), 3. REINFORCE with Baseline, Action Space, AUTODSPy, Citations, Common Components, Common Issues (+20 more)

### Community 4 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 5 - "test_reproduction.py"
Cohesion: 0.08
Nodes (14): Config, Example, action_loss(), group_advantages(), group_loss(), seed_everything(), update_episode(), fixture_policy() (+6 more)

### Community 6 - "Execution"
Cohesion: 0.22
Nodes (4): Execution, pipeline_pair(), ScriptedExecutionFixture, FixtureExecutor

### Community 7 - "Policy"
Cohesion: 0.22
Nodes (3): Decision, Policy, Trajectory

### Community 8 - "📊 CSE 498R Thesis: Baseline Execution & Empirical Analysis Report"
Cohesion: 0.09
Nodes (21): 1. Executive Summary & The Core Problem, 2. Experimental Environment & Hardware Architecture, 3. The GRPO Mathematical Framework in AutoDSPy, 4. Run 1: Untouched Author Baseline (Sanity Verification), 5. Run 2: Formal Static $K=4$ Baseline (30 Episodes), 6. Complete 30-Episode Trajectory Log, 7. In-Depth Empirical Analysis & Failure Taxonomy, 8. The Thesis Solution: Adaptive-$k$ Two-Stage Probe (+13 more)

### Community 9 - "run_grpo_baseline.py"
Cohesion: 0.21
Nodes (7): Core Functions, compare_answers(), compute_reward(), execute_pipeline(), generate_pipeline(), main(), train_grpo_baseline()

### Community 10 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 11 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 12 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 13 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 14 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 20 - "__main__.py"
Cohesion: 0.12
Nodes (11): load_config(), check_no_leakage(), load_examples(), DSPyExecutor, ollama_json(), preflight(), main(), action_distribution() (+3 more)

## Knowledge Gaps
- **102 isolated node(s):** `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)`, `Step 1 - Ensure graphify is installed`, `Step 2 - Detect files` (+97 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 169 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train()` connect `runtime.py` to `Findings and proposed corrections`, `__main__.py`, `test_reproduction.py`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `Model and dataset setup` connect `Findings and proposed corrections` to `runtime.py`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Config` (e.g. with `PolicyTests` and `RunTests`) actually correct?**
  _`Config` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` to the rest of the system?**
  _102 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `reward.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13725490196078433 - nodes in this community are weakly interconnected._
- **Should `Findings and proposed corrections` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._
- **Should `DSPy Pipeline Optimization with Reinforcement Learning` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._