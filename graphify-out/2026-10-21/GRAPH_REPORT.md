# Graph Report - AUTODSPy  (2026-10-21)

## Corpus Check
- 379 files · ~837,953 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 129 file(s) not represented in the graph (top: .jsonl 80, .csv 18, .ipynb 15)

## Summary
- 833 nodes · 858 edges · 63 communities (50 shown, 13 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `595cc96a`
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
- AdaGRPO GSM8K implementation and verification
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
- Additional Papers for Adaptive-K in AutoDSPy
- <Questionnaire title>
- writing-shape/SKILL.md
- Process
- Sections
- writing-beats/SKILL.md
- Adaptive-K for AutoDSPy: evidence, current status, and research objective
- loop-me/SKILL.md
- Reference
- Domain docs
- Issue tracker: GitHub
- chief-of-staff/SKILL.md
- hitl-loop.template.sh
- GLOSSARY.md Format
- writing-fragments/SKILL.md
- Project progress and session handoff
- Static-K experiment guide
- AdaGRPO V3 implementation verification
- block-dangerous-git.sh
- implement-spec/SKILL.md
- CREDITS.md
- agents/triage-labels.md
- Completed K=5 GSM8K run diagnosis
- K=5 controlled stability pilot
- Static-K GRPO on GSM8K: saved experimental results
- Process

## God Nodes (most connected - your core abstractions)
1. `Decisions, assumptions, and open research questions` - 45 edges
2. `AutoDSPy Static-K Research — Master Specification` - 31 edges
3. `Project progress and session handoff` - 13 edges
4. `run_wizard()` - 12 edges
5. `Additional Papers for Adaptive-K in AutoDSPy` - 12 edges
6. `DSPy Pipeline Optimization with Reinforcement Learning` - 11 edges
7. `Static-K experiment guide` - 11 edges
8. `Adaptive-K for AutoDSPy: evidence, current status, and research objective` - 11 edges
9. `Static-K GRPO on GSM8K: saved experimental results` - 11 edges
10. `main()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `5. Reward consistency: what exists and what remains` --references--> `compute_reward()`  [INFERRED]
  docs/ADAPTIVE_K_RESEARCH_UNDERSTANDING.md → run_grpo_baseline.py
- `1. Pin the fixed point` --references--> `main()`  [INFERRED]
  .agents/skills/code-review/SKILL.md → run_grpo_baseline.py
- `1. Experimental configuration and comparability audit` --references--> `main()`  [INFERRED]
  docs/STATIC_K_RESULTS_ANALYSIS.md → run_grpo_baseline.py
- `Answer matching produces false negatives and false positives` --references--> `compare_answers()`  [INFERRED]
  docs/PHASE_1_AUDIT.md → run_grpo_baseline.py
- `Reward semantics and failure handling` --references--> `compute_reward()`  [INFERRED]
  docs/PHASE_1_AUDIT.md → run_grpo_baseline.py

## Import Cycles
- None detected.

## Communities (63 total, 13 thin omitted)

### Community 0 - "run_grpo_baseline.py"
Cohesion: 0.06
Nodes (37): 6. Preserve the Original DSPy Action Space, Checked-in reference notebook: actual behavior and caveats, Implication for the current project, Original AutoDSPy paper and checked-in reference notebook, Paper ambiguities to preserve in any reproduction claim, Published experiment setup and results, Source identity and citation correction, The original paper already studies fixed K and proposes Adaptive-K (+29 more)

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
Cohesion: 0.06
Nodes (34): 1. PPO (Proximal Policy Optimization), 2. GRPO (Group Relative Policy Optimization), 3. REINFORCE with Baseline, Action Space, AdaGRPO research fork, AdaGRPO V3: early exploration, AdaGRPO version 2, Citations (+26 more)

### Community 7 - "Process"
Cohesion: 0.07
Nodes (25): 1. State the question, 2. Isolate the logic in a portable module, 3. Build the shareable HTML file, 4. Hand it over, 5. Capture the answer and the prototype, Anti-patterns, Logic Prototype, Process (+17 more)

### Community 8 - "AdaGRPO GSM8K implementation and verification"
Cohesion: 0.05
Nodes (35): 200-episode GSM8K study, 30-episode GSM8K study, Ahnaf's GSM8K Static-K Results: How AdaGRPO Should Use Them, Completed checkpoint evaluation (fresh 100), Purpose, Supplied sources, What this contributes to AdaGRPO, AdaGRPO GSM8K implementation and verification (+27 more)

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
Cohesion: 0.04
Nodes (45): D001 — Prepare documents, then request Phase 1 permission, D002 — Store persistent context in docs/, D003 — Preserve the original and existing working-tree changes, D004 — Figure 2 objective and episode protocol are fixed, D005 — Independent notebooks and model initialization, D006 — Configurable defaults and reduced engineering tests, D007 — Benchmark splits and unresolved HotpotQA formulation, D008 — Reward, benchmark metrics, and model failures stay distinct (+37 more)

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

### Community 28 - "Additional Papers for Adaptive-K in AutoDSPy"
Cohesion: 0.12
Nodes (15): 1. AERO: probe first, then give bounded extra attempts, 1. Reinforce-Ada, 2. GVM, 2. Reinforce-Ada: make the stopping condition explicit, 3. VIGOR, 4. DPPO, Added reference: GRPO-LEAD, Additional Papers for Adaptive-K in AutoDSPy (+7 more)

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

### Community 34 - "Adaptive-K for AutoDSPy: evidence, current status, and research objective"
Cohesion: 0.12
Nodes (15): 10. Evidence boundaries and housekeeping, 1. The central objective, 2. What the original system actually learns, 3. Original paper versus our experiments, 4. What the team has completed, 5. Reward consistency: what exists and what remains, 6. Important earlier work on other branches, 7. What Adaptive-K needs to decide (+7 more)

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

### Community 43 - "Project progress and session handoff"
Cohesion: 0.15
Nodes (13): Completed Phase 1 work, Completed Phase 2 work and verification, Completed Phase 3 (2026-10-12), Completed Phase 4 (2026-10-12), Completed preparation, Current milestone, Pending work and unresolved questions, Phase status (+5 more)

### Community 44 - "Static-K experiment guide"
Cohesion: 0.18
Nodes (11): A K=5 smoke run, Changing K, budgets and presets, Comparisons, plots and resource accounting, Environment and initialization, HotpotQA configuration and use, Monitoring GPU memory and interpreting evidence, Other K values and presets, Questions, episodes and results (+3 more)

### Community 45 - "AdaGRPO V3 implementation verification"
Cohesion: 0.33
Nodes (5): AdaGRPO V3 implementation verification, Authorized live study, Before live execution, Completed local screen, Evidence

### Community 63 - "Completed K=5 GSM8K run diagnosis"
Cohesion: 0.20
Nodes (9): All episodes, Completed K=5 GSM8K run diagnosis, Finding, Fractional-mean groups, Gradient and optimizer checks, Other observed issues and limits, Probability and entropy evidence, Reproduction and preservation (+1 more)

### Community 65 - "K=5 controlled stability pilot"
Cohesion: 0.33
Nodes (5): Controls, K=5 controlled stability pilot, Measured stability, Rewards and outcomes, Verification and preservation

### Community 66 - "Static-K GRPO on GSM8K: saved experimental results"
Cohesion: 0.15
Nodes (12): 10. Recommended next research experiment, 1. Experimental configuration and comparability audit, 2. Summary comparing all four K values, 3. Model performance analysis, 4. GRPO learning behavior and group effectiveness, 5. Pipeline diversity analysis, 6. Training stability and reward integrity, 7. Computational efficiency comparison (+4 more)

### Community 67 - "Process"
Cohesion: 0.25
Nodes (7): 1. Pin the fixed point, 2. Identify the spec source, 3. Identify the standards sources, 4. Spawn both sub-agents in parallel, 5. Aggregate, Process, Why two axes

## Knowledge Gaps
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AutoDSPy Static-K Research — Master Specification` connect `AutoDSPy Static-K Research — Master Specification` to `run_grpo_baseline.py`, `IMPLEMENTATION_PLAN.md`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Should `run_grpo_baseline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06462585034013606 - nodes in this community are weakly interconnected._
- **Why does `Decisions, assumptions, and open research questions` connect `Decisions, assumptions, and open research questions` to `IMPLEMENTATION_PLAN.md`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Should `AutoDSPy Static-K Research — Master Specification` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._
- **Why does `DSPy Pipeline Optimization with Reinforcement Learning` connect `DSPy Pipeline Optimization with Reinforcement Learning` to `AdaGRPO GSM8K implementation and verification`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Should `Issue tracker: GitHub` be split into smaller, more focused modules?**
  _Cohesion score 0.05555555555555555 - nodes in this community are weakly interconnected._
- **Should `IMPLEMENTATION_PLAN.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0728744939271255 - nodes in this community are weakly interconnected._