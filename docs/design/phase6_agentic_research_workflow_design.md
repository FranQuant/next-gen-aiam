# Phase 6 Agentic Research Workflow Design

## 1. Objective

Phase 6 defines a governed agentic research workflow for asset management. The
workflow consumes deterministic repository artifacts and produces a structured
research committee packet for human review.

Phase 6 is not an autonomous portfolio manager. It does not make allocation
decisions, place trades, route orders, or present outputs as investment advice.
Its purpose is to organize existing evidence, identify interpretation gaps, and
make review boundaries explicit.

This memo covers Phase 6A design-only work. Implementation of Python code,
tests, schemas, manifests, validators, or orchestration wrappers is out of scope
for this phase.

## 2. Scope

Phase 6A scopes the design for:

- A deterministic artifact read surface.
- Framework-neutral artifact readers and tool contracts.
- Research packet schemas and validation rules.
- Artifact traceability from packet claims back to repository files.
- Human review gates before any downstream research or investment process can
  use the packet.
- A later orchestration layer that may wrap the deterministic core.

The workflow must operate on repository artifacts already produced by prior
research, validation, and reporting steps.

## 3. Non-goals

Phase 6A does not:

- Implement code, tests, manifests, schemas, validators, or orchestration
  wrappers.
- Train, tune, or evaluate models during packet generation.
- Introduce an autonomous trading or portfolio management system.
- Create new allocation targets, recommended portfolio weights, or trade lists.
- Connect to live market data, news, brokerage APIs, custodians, or order
  routers.
- Modify notebooks, source files, data files, results, or published artifacts.
- Replace human research committee review.

## 4. Book alignment

The workflow is inspired by Chapter 23 of Hilpisch's *Python and AI for Asset
Management*, with the important constraint that this repository treats agentic
systems as governed research assistants rather than autonomous decision makers.

The design aligns with the book's agentic research theme by decomposing work
into evidence gathering, interpretation, critique, and synthesis. It diverges
from any fully autonomous trading interpretation by requiring deterministic
inputs, explicit validation, artifact traceability, and human review.

## 5. Repository alignment

The Phase 6 design should build on existing repository patterns rather than
introducing a framework-first architecture. Relevant existing surfaces include:

- `src/aiam/llm/schemas.py`
- `src/aiam/llm/evidence.py`
- `src/aiam/llm/prompts.py`
- `src/aiam/llm/views.py`
- `src/aiam/evaluation/performance.py`
- `src/aiam/evaluation/regime_conditional.py`
- `src/aiam/evaluation/transaction_costs.py`
- `src/aiam/evaluation/ic.py`
- `docs/research/`
- `docs/validation/`
- `docs/execution_packets/`
- `results/notebook_07/`
- `results/notebook_07b/`
- `data/published/`

The completed read-only Phase 6 audit was performed on branch
`agents/phase6-design`, and the test suite passed with `370 passed`. Those
findings support a conservative design path: deterministic artifact readers,
schemas, validators, and manifests should come before agent orchestration.

## 6. Approved read surface

The Phase 6 workflow may read only artifacts included in an approved manifest.
The approved read surface is:

- `data/published/`
- `results/notebook_07/`
- `results/notebook_07b/`
- Selected `results/notebook_05/` summary files.
- Selected `results/cuda/` summary files.
- Selected `results/llm/` diagnostics, returns, and weights only as historical
  experiment artifacts.
- `docs/results.md`
- `docs/results.pdf`
- `docs/research/pca_dislocation_macro_momentum_report_review.md`
- `docs/validation/*.md`
- `docs/design/*.md`

Selection of partial result surfaces must be represented in the approved
artifact manifest. Direct reads outside the manifest are not permitted.

## 7. Explicitly excluded surfaces

The Phase 6 workflow must not read:

- `.env`
- `.claude/`
- `.venv/`
- `data/cache/llm/*.json`
- `notebooks/.ipynb_checkpoints/`
- `notebooks/_rl_dev/`
- `results/rl/**/agent_seed_*.pt`
- `__pycache__`
- `.pytest_cache`
- Arbitrary filesystem paths not included in the approved manifest.

Excluded surfaces protect credentials, private caches, development artifacts,
checkpoint files, and non-reviewed filesystem content from entering the
research packet.

## 8. Allowed capabilities

The workflow may:

- Read the approved artifact manifest.
- List approved artifacts.
- Load deterministic research, validation, and result summaries from approved
  files.
- Summarize backtest metrics from existing artifacts.
- Summarize regime-conditional signal evidence from existing artifacts.
- Summarize PCA dashboard evidence from existing artifacts.
- Build a structured research committee packet.
- Validate packet structure, citations, provenance, and review status.
- Identify missing evidence, stale artifacts, inconsistent claims, and required
  human decisions.

All allowed capabilities are read-only with respect to repository artifacts.

## 9. Forbidden capabilities

The workflow must not:

- Place trades.
- Route orders.
- Connect to brokerage APIs.
- Produce target portfolio weights as recommendations.
- Change allocation decisions.
- Train models during packet generation.
- Modify notebooks, source files, data, or results.
- Read credentials or private caches.
- Use live web or news data as direct allocation input.
- Present outputs as investment advice.

These prohibitions apply to the deterministic core and to any later
orchestration wrapper.

## 10. Workflow architecture

The architecture should be implemented in this order in a later phase:

1. Define an approved artifact manifest that enumerates readable files,
   artifact types, expected checksums or timestamps, and permitted parsers.
2. Implement deterministic artifact readers for approved markdown, PDF,
   published data, and result summary artifacts.
3. Define packet schemas that separate evidence, interpretation, caveats, gaps,
   and human review fields.
4. Implement validators for manifest compliance, citation coverage, schema
   completeness, and forbidden-capability checks.
5. Build a deterministic packet generation path from approved artifacts to a
   structured research committee packet.
6. Add optional orchestration wrappers only after the deterministic core is
   stable and validated.

The core must remain framework-neutral. Agent frameworks may orchestrate calls
to the deterministic tool surface, but they must not own the research logic,
artifact policy, validation policy, or packet schema.

## 11. Agent roles

Later orchestration may use specialized roles, each constrained to approved
tools:

- Evidence collector: enumerates approved artifacts and extracts cited facts.
- Metrics analyst: summarizes existing backtest, transaction-cost, performance,
  and IC evidence.
- Regime analyst: summarizes existing regime-conditional signal evidence.
- PCA dashboard analyst: summarizes existing PCA dislocation and macro momentum
  evidence.
- Skeptic: identifies unsupported claims, missing citations, stale artifacts,
  horizon mismatches, leakage risks, and validation gaps.
- Packet editor: assembles the committee packet from validated evidence and
  unresolved questions.

Agents may interpret approved evidence, but they may not expand the read
surface, generate new research artifacts, or convert observations into
investment advice.

## 12. Framework-neutral tool surface

The deterministic tool surface should expose these framework-neutral operations:

- `read_artifact_manifest`
- `list_approved_artifacts`
- `summarize_backtest_metrics`
- `summarize_regime_signal`
- `summarize_pca_dashboard`
- `build_research_packet`
- `validate_research_packet`

Each operation should be callable from tests, scripts, notebooks, or an agent
orchestration wrapper without requiring any specific agent framework.

Agno's `example8.py` may be used later as the first orchestration wrapper
because it follows an evidence, interpretation, and gaps workflow that matches
the intended review process. OpenAI Agents SDK remains a possible later variant,
but it should not be the first dependency or the owner of the core design.

## 13. Output schema

The research committee packet should include, at minimum:

- Packet metadata: packet ID, generation time, repository branch, manifest ID,
  artifact manifest version, and validation status.
- Executive research summary: concise, research-only synthesis.
- Evidence inventory: approved artifacts read, artifact type, path, checksum or
  timestamp, and parser used.
- Backtest metrics summary: cited performance, drawdown, turnover,
  transaction-cost, and robustness evidence from approved artifacts.
- Regime signal summary: cited regime-conditional behavior and limitations.
- PCA dashboard summary: cited dislocation, macro momentum, and dashboard
  evidence.
- Historical experiment context: clearly labeled diagnostics, returns, and
  weights from selected `results/llm/` artifacts when included, without
  presenting them as recommendations.
- Interpretations: separated from evidence and linked to supporting citations.
- Caveats: methodology concerns, data limitations, execution assumptions,
  horizon mismatches, leakage risks, and transaction-cost sensitivity.
- Open questions: unresolved issues requiring human review.
- Forbidden-content check: explicit confirmation that the packet contains no
  trades, order-routing instructions, target portfolio recommendations, or
  investment advice.
- Human review fields: reviewer, review date, decision, required follow-ups,
  and approval status.

The schema should make unsupported claims invalid rather than merely
discouraged.

## 14. Artifact traceability

Every material claim in the packet must trace back to an approved artifact.
Traceability should include:

- Artifact path.
- Artifact type.
- Section, table, row, page, or record reference when available.
- Artifact timestamp, checksum, or version marker.
- Parser or reader used to extract the evidence.

Derived summaries must identify the source artifacts and calculation assumptions.
If a claim cannot be traced, the packet must either omit it or list it as an
open question.

## 15. Validation rules

Packet validation should fail when:

- Any read artifact is absent from the approved manifest.
- Any excluded path is accessed or cited.
- Required packet sections are missing.
- Evidence and interpretation are not separated.
- A material claim lacks artifact traceability.
- Historical experiment artifacts are presented as current recommendations.
- The packet contains target portfolio weights as recommendations.
- The packet includes trade instructions, order-routing instructions, or
  brokerage connectivity.
- The packet uses live web or news data as direct allocation input.
- The packet modifies, or claims to modify, notebooks, source files, data, or
  results.
- Human review fields are missing from the final packet.

Validation should also warn on stale artifacts, inconsistent metrics,
unexplained horizon mismatches, missing transaction-cost context, and ambiguous
execution timing assumptions.

## 16. Human review boundary

All Phase 6 outputs are research-only and require human review. A generated
packet may support discussion by a research committee, but it cannot approve,
reject, resize, rebalance, or execute any allocation.

Any downstream use must be gated by explicit human review fields in the packet.
The workflow should make unresolved questions and required follow-ups visible
rather than smoothing them into a false recommendation.

## 17. Implementation phases

Later implementation should proceed in small, reviewable phases:

1. Phase 6B: define the artifact manifest format and approved artifact
   inventory.
2. Phase 6C: implement deterministic artifact readers for the approved read
   surface.
3. Phase 6D: define packet schemas and validation rules.
4. Phase 6E: implement deterministic packet generation and validation.
5. Phase 6F: add focused tests for manifest enforcement, traceability,
   validation failures, and research-only boundaries.
6. Phase 6G: add an optional Agno wrapper, potentially based on the
   `example8.py` evidence, interpretation, and gaps pattern.
7. Phase 6H: evaluate whether OpenAI Agents SDK is useful as a later variant
   without changing the framework-neutral core.

No implementation work belongs in Phase 6A.

## 18. Acceptance criteria

Phase 6A is complete when:

- This memo exists as the single Phase 6 design memo.
- The memo states that Phase 6 is a governed research workflow, not an
  autonomous portfolio manager.
- The approved read surface and explicitly excluded surfaces are documented.
- Allowed and forbidden capabilities are documented.
- The deterministic, framework-neutral tool surface is defined.
- The architecture puts artifact readers, schemas, validators, and manifests
  before orchestration.
- Agno is positioned as a possible first wrapper and OpenAI Agents SDK as a
  possible later variant.
- Output schema, artifact traceability, validation rules, and human review
  boundaries are documented.
- No Python code, tests, manifests, schemas, or wrappers are added.

## 19. Risks and open issues

- Manifest design must be strict enough to prevent arbitrary filesystem access
  while still covering required research artifacts.
- PDF and markdown extraction need deterministic citation references that are
  stable enough for review.
- Selected partial surfaces such as `results/notebook_05/`, `results/cuda/`,
  and `results/llm/` need explicit file-level manifest entries.
- Historical weights in `results/llm/` are especially sensitive and must remain
  labeled as historical experiment artifacts, not recommendations.
- Packet validators need clear rules for separating evidence from
  interpretation.
- Human review workflow ownership remains to be defined.
- Orchestration wrappers may create pressure to move policy into framework
  code; the deterministic core should remain the source of truth.
