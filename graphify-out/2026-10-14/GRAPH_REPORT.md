# Graph Report - AUTODSPy  (2026-10-13)

## Corpus Check
- 131 files · ~178,862 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .jsonl 9, .ipynb 6, (none) 4)

## Summary
- 773 nodes · 772 edges · 66 communities (50 shown, 16 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4432567c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run_grpo_baseline.py
- AutoDSPy Static-K Research — Master Specification
- Issue tracker: GitHub
- IMPLEMENTATION_PLAN.md
- Triage
- teach/SKILL.md
- DSPy Pipeline Optimization with Reinforcement Learning
- Process
- What You Must Do When Invoked
- Codebase Design
- During the session
- 📊 CSE 498R Thesis: Baseline Execution & Empirical Analysis Report
- HTML Report Format
- template.sh
- Decisions, assumptions, and open research questions
- Ask Matt
- AutoDSPy Research — Agent Instructions
- Diagnosing Bugs
- Steps
- Test-Driven Development
- Process
- writing-for-agents/SKILL.md
- Steps
- wayfinder/SKILL.md
- Migrate to Shoehorn
- Steps
- Scaffold Exercises
- to-spec/SKILL.md
- graphify reference: extra exports and benchmark
- <Questionnaire title>
- writing-shape/SKILL.md
- Process
- Sections
- writing-beats/SKILL.md
- graphify reference: query, path, explain
- loop-me/SKILL.md
- Reference
- Domain docs
- Issue tracker: GitHub
- chief-of-staff/SKILL.md
- hitl-loop.template.sh
- GLOSSARY.md Format
- writing-fragments/SKILL.md
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- block-dangerous-git.sh
- implement-spec/SKILL.md
- CREDITS.md
- extraction-spec.md
- agents/triage-labels.md
- Static-K experiment guide
- Completed K=5 GSM8K run diagnosis
- Project progress and session handoff
- K=5 controlled stability pilot

## God Nodes (most connected - your core abstractions)
1. `Decisions, assumptions, and open research questions` - 31 edges
2. `AutoDSPy Static-K Research — Master Specification` - 31 edges
3. `Project progress and session handoff` - 13 edges
4. `run_wizard()` - 12 edges
5. `What You Must Do When Invoked` - 12 edges
6. `Static-K experiment guide` - 11 edges
7. `/graphify` - 10 edges
8. `AutoDSPy Research — Agent Instructions` - 10 edges
9. `📊 CSE 498R Thesis: Baseline Execution & Empirical Analysis Report` - 10 edges
10. `DSPy Pipeline Optimization with Reinforcement Learning` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Reward semantics and failure handling` --references--> `compute_reward()`  [INFERRED]
  docs/PHASE_1_AUDIT.md → run_grpo_baseline.py
- `Evaluation can fail or use the wrong policy` --references--> `generate_pipeline()`  [INFERRED]
  docs/PHASE_1_AUDIT.md → run_grpo_baseline.py
- `Initialization and gradient ownership` --references--> `generate_pipeline()`  [INFERRED]
  docs/PHASE_1_AUDIT.md → run_grpo_baseline.py
- `1. Pin the fixed point` --references--> `main()`  [INFERRED]
  .agents/skills/code-review/SKILL.md → run_grpo_baseline.py
- `7. Notebook A — GSM8K` --references--> `main()`  [INFERRED]
  docs/MASTER_SPEC.md → run_grpo_baseline.py

## Import Cycles
- None detected.

## Communities (66 total, 16 thin omitted)

### Community 0 - "run_grpo_baseline.py"
Cohesion: 0.05
Nodes (36): 1. Pin the fixed point, 2. Identify the spec source, 3. Identify the standards sources, 4. Spawn both sub-agents in parallel, 5. Aggregate, Process, Why two axes, 6. Preserve the Original DSPy Action Space (+28 more)

### Community 1 - "AutoDSPy Static-K Research — Master Specification"
Cohesion: 0.05
Nodes (44): 10. Independent K Experiments, 11. Training Correctness, 12. Checkpointing and Resuming, 13. Validation and Testing, 14. Research Statistics, 15. Visualizations, 16. Simple Notebook Execution Interface, 17. Notebook Organization (+36 more)

### Community 2 - "Issue tracker: GitHub"
Cohesion: 0.06
Nodes (30): Before exploring, read these, Domain Docs, File structure, Flag ADR conflicts, Use the glossary's vocabulary, Conventions, Issue tracker: GitHub, Pull requests as a triage surface (+22 more)

### Community 3 - "IMPLEMENTATION_PLAN.md"
Cohesion: 0.07
Nodes (29): Phase 1 audit artifacts, Phase 2 verification additions, Phase 3 verification additions, Execution rules and session handoff, Phase 1 — Audit original notebook and research paper, Phase 2 — Implement and validate the complete GSM8K notebook, Phase 3 — Implement and validate the independent HotpotQA notebook, Phase 4 — Finalize management, statistics, plots, and reproducibility (+21 more)

### Community 4 - "Triage"
Cohesion: 0.06
Nodes (29): Bad agent brief, Behavioral, not procedural, Complete acceptance criteria, Durability over precision, Examples, Explicit scope boundaries, Good agent brief (bug), Good agent brief (enhancement) (+21 more)

### Community 5 - "teach/SKILL.md"
Cohesion: 0.07
Nodes (25): Learning Record Format, Numbering, Optional sections, Supersession, Template, What does _not_ qualify, When to write a learning record, MISSION.md Format (+17 more)

### Community 6 - "DSPy Pipeline Optimization with Reinforcement Learning"
Cohesion: 0.07
Nodes (28): 1. PPO (Proximal Policy Optimization), 2. GRPO (Group Relative Policy Optimization), 3. REINFORCE with Baseline, Action Space, AUTODSPy, Citations, Common Components, Common Issues (+20 more)

### Community 7 - "Process"
Cohesion: 0.07
Nodes (25): 1. State the question, 2. Isolate the logic in a portable module, 3. Build the shareable HTML file, 4. Hand it over, 5. Capture the answer and the prototype, Anti-patterns, Logic Prototype, Process (+17 more)

### Community 8 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 9 - "Codebase Design"
Cohesion: 0.09
Nodes (21): 1. In-process, 2. Local-substitutable, 3. Remote but owned (Ports & Adapters), 4. True external (Mock), Deepening, Dependency categories, Seam discipline, Testing strategy: replace, don't layer (+13 more)

### Community 10 - "During the session"
Cohesion: 0.09
Nodes (19): ADR Format, Numbering, Optional sections, Template, What qualifies, When to offer an ADR, GLOSSARY.md Format, Rules (+11 more)

### Community 11 - "📊 CSE 498R Thesis: Baseline Execution & Empirical Analysis Report"
Cohesion: 0.09
Nodes (21): 1. Executive Summary & The Core Problem, 2. Experimental Environment & Hardware Architecture, 3. The GRPO Mathematical Framework in AutoDSPy, 4. Run 1: Untouched Author Baseline (Sanity Verification), 5. Run 2: Formal Static $K=4$ Baseline (30 Episodes), 6. Complete 30-Episode Trajectory Log, 7. In-Depth Empirical Analysis & Failure Taxonomy, 8. The Thesis Solution: Adaptive-$k$ Two-Stage Probe (+13 more)

### Community 12 - "HTML Report Format"
Cohesion: 0.10
Nodes (18): Call-graph collapse, Candidate card, Cross-section (good for layered shallowness), Diagram patterns, Hand-built boxes-and-arrows (when Mermaid's layout fights you), Header, HTML Report Format, Mass diagram (good for "interface as wide as implementation") (+10 more)

### Community 13 - "template.sh"
Cohesion: 0.22
Nodes (18): ask(), ask_secret(), banner(), _clear(), _existing(), finish(), note(), open_url() (+10 more)

### Community 14 - "Decisions, assumptions, and open research questions"
Cohesion: 0.06
Nodes (31): D001 — Prepare documents, then request Phase 1 permission, D002 — Store persistent context in docs/, D003 — Preserve the original and existing working-tree changes, D004 — Figure 2 objective and episode protocol are fixed, D005 — Independent notebooks and model initialization, D006 — Configurable defaults and reduced engineering tests, D007 — Benchmark splits and unresolved HotpotQA formulation, D008 — Reward, benchmark metrics, and model failures stay distinct (+23 more)

### Community 15 - "Ask Matt"
Cohesion: 0.12
Nodes (14): Phase boundaries, Primary and secondary sources, The five options, The tree, These are judgement calls, Ask Matt, Codebase health, Context hygiene (+6 more)

### Community 16 - "AutoDSPy Research — Agent Instructions"
Cohesion: 0.13
Nodes (14): Agent skills, Architecture Rules, AutoDSPy Research — Agent Instructions, Development Workflow, Domain docs, Experimental Rules, graphify, GRPO Rules (+6 more)

### Community 17 - "Diagnosing Bugs"
Cohesion: 0.13
Nodes (14): Completion criterion: a tight loop that goes red, Diagnosing Bugs, Minimise, Non-deterministic bugs, Phase 1: Build a feedback loop, Phase 2: Reproduce + minimise, Phase 3: Hypothesise, Phase 4: Instrument (+6 more)

### Community 18 - "Steps"
Cohesion: 0.15
Nodes (12): 1. Detect package manager, 2. Install dependencies, 3. Initialize Husky, 4. Create `.husky/pre-commit`, 5. Create `.lintstagedrc`, 6. Create `.prettierrc` (if missing), 7. Verify, 8. Commit (+4 more)

### Community 19 - "Test-Driven Development"
Cohesion: 0.15
Nodes (10): Designing for Mockability, When to Mock, Anti-patterns, Rules of the loop, Seams: where tests go, Test-Driven Development, What a good test is, Bad Tests (+2 more)

### Community 20 - "Process"
Cohesion: 0.15
Nodes (12): 1. Gather context, 2. Explore the codebase (optional), 3. Draft vertical slices, 4. Quiz the user, 5. Publish the tickets to the configured tracker, Acceptance criteria, Blocked by, <NN>: <Ticket title> (+4 more)

### Community 21 - "writing-for-agents/SKILL.md"
Cohesion: 0.15
Nodes (11): Context pointers, Information hierarchy, Leading words, Invocation, Router skills, Skill mechanics, Splitting by invocation, Pruning (+3 more)

### Community 22 - "Steps"
Cohesion: 0.17
Nodes (11): 1. Detect the environment, 2. Install dependency-cruiser, 3. Write the config, 4. Wire it into the checks, 5. Scaffold the example package, 6. Prove the rules bite, 7. Document the convention, Notes (+3 more)

### Community 23 - "wayfinder/SKILL.md"
Cohesion: 0.17
Nodes (11): Chart the map, Fog of war, Invocation, Out of scope, Plan, don't do, Refer by name, The Map, The map body (+3 more)

### Community 24 - "Migrate to Shoehorn"
Cohesion: 0.20
Nodes (9): `as Type` → `fromPartial()`, `as unknown as Type` → `fromAny()`, Install, Large objects with few needed properties, Migrate to Shoehorn, Migration patterns, When to use each, Why shoehorn? (+1 more)

### Community 25 - "Steps"
Cohesion: 0.22
Nodes (8): 1. Ask scope, 2. Copy the hook script, 3. Add hook to settings, 4. Ask about customization, 5. Verify, Setup Git Guardrails, Steps, What Gets Blocked

### Community 26 - "Scaffold Exercises"
Cohesion: 0.22
Nodes (8): Directory naming, Example: stubbing from a plan, Exercise variants, Lint rules summary, Moving/renaming exercises, Required files, Scaffold Exercises, Workflow

### Community 27 - "to-spec/SKILL.md"
Cohesion: 0.22
Nodes (8): Further Notes, Implementation Decisions, Out of Scope, Problem Statement, Process, Solution, Testing Decisions, User Stories

### Community 28 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 29 - "<Questionnaire title>"
Cohesion: 0.25
Nodes (7): Anything else?, Context, Document structure, How to answer, <Questionnaire title>, <Theme heading>, What load is the system expected to handle at launch?

### Community 30 - "writing-shape/SKILL.md"
Cohesion: 0.25
Nodes (7): Conversational feel, Format arguments to actually have, Grounding, Out of scope, Pulling from the pile, The loop, Writing rhythm

### Community 31 - "Process"
Cohesion: 0.29
Nodes (6): 1. Scope the procedure, 2. Map each stage's journey, 3. Author the wizard, 4. Verify and hand off, Process, Wizard

### Community 32 - "Sections"
Cohesion: 0.33
Nodes (5): Evidence, Guidance, Merge Danger, Sections, Summary

### Community 33 - "writing-beats/SKILL.md"
Cohesion: 0.33
Nodes (5): Ending the journey, Grounding, Pulling from the pile, What is a beat, Writing rhythm

### Community 34 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 35 - "loop-me/SKILL.md"
Cohesion: 0.40
Nodes (4): Definition of done, The loop lens, The workspace, Vocabulary

### Community 36 - "Reference"
Cohesion: 0.40
Nodes (4): Files, Implementation vs Review, Reference, Steps

### Community 37 - "Domain docs"
Cohesion: 0.40
Nodes (4): Decision conflicts, Domain docs, Layout and reading rules, Vocabulary

### Community 38 - "Issue tracker: GitHub"
Cohesion: 0.40
Nodes (4): Issue tracker: GitHub, Operations, Pull requests as a triage surface, Skill instructions

### Community 39 - "chief-of-staff/SKILL.md"
Cohesion: 0.50
Nodes (3): Schedules, Strategic View, Subagents

### Community 40 - "hitl-loop.template.sh"
Cohesion: 0.83
Nodes (3): capture(), hitl-loop.template.sh script, step()

### Community 41 - "GLOSSARY.md Format"
Cohesion: 0.50
Nodes (3): GLOSSARY.md Format, Rules, Structure

### Community 42 - "writing-fragments/SKILL.md"
Cohesion: 0.50
Nodes (3): File format, What is a fragment, Writing rhythm

### Community 43 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 44 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 45 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 62 - "Static-K experiment guide"
Cohesion: 0.18
Nodes (11): A K=5 smoke run, Changing K, budgets and presets, Comparisons, plots and resource accounting, Environment and initialization, HotpotQA configuration and use, Monitoring GPU memory and interpreting evidence, Other K values and presets, Questions, episodes and results (+3 more)

### Community 63 - "Completed K=5 GSM8K run diagnosis"
Cohesion: 0.20
Nodes (9): All episodes, Completed K=5 GSM8K run diagnosis, Finding, Fractional-mean groups, Gradient and optimizer checks, Other observed issues and limits, Probability and entropy evidence, Reproduction and preservation (+1 more)

### Community 64 - "Project progress and session handoff"
Cohesion: 0.15
Nodes (13): Completed Phase 1 work, Completed Phase 2 work and verification, Completed Phase 3 (2026-10-12), Completed Phase 4 (2026-10-12), Completed preparation, Current milestone, Pending work and unresolved questions, Phase status (+5 more)

### Community 65 - "K=5 controlled stability pilot"
Cohesion: 0.33
Nodes (5): Controls, K=5 controlled stability pilot, Measured stability, Rewards and outcomes, Verification and preservation

## Knowledge Gaps
- **513 isolated node(s):** `block-dangerous-git.sh script`, `The five options`, `The tree`, `Primary and secondary sources`, `These are judgement calls` (+508 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 562 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AutoDSPy Static-K Research — Master Specification` connect `AutoDSPy Static-K Research — Master Specification` to `run_grpo_baseline.py`, `IMPLEMENTATION_PLAN.md`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `Decisions, assumptions, and open research questions` connect `Decisions, assumptions, and open research questions` to `IMPLEMENTATION_PLAN.md`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `6. Preserve the Original DSPy Action Space` connect `run_grpo_baseline.py` to `AutoDSPy Static-K Research — Master Specification`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **What connects `block-dangerous-git.sh script`, `The five options`, `The tree` to the rest of the system?**
  _513 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run_grpo_baseline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05272108843537415 - nodes in this community are weakly interconnected._
- **Should `AutoDSPy Static-K Research — Master Specification` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._
- **Should `Issue tracker: GitHub` be split into smaller, more focused modules?**
  _Cohesion score 0.05555555555555555 - nodes in this community are weakly interconnected._