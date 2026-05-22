# Phase 6B Artifact Manifest Specification

## 1. Purpose

Phase 6B defines the first approved artifact manifest for the governed Phase 6
research workflow. The manifest is a deterministic, file-level policy document
that states which repository artifacts may be read when building a structured
research committee packet for human review.

The manifest does not implement agent orchestration, model calls, live data
access, portfolio construction, trade generation, or investment advice. It only
defines the permitted evidence surface for later deterministic tools.

## 2. Explicit allowlisting

The manifest uses `forbidden_default: true`. Any file not listed under
`approved_artifacts` is forbidden, even if it sits near an approved artifact.
This is required because Phase 6 consumes research evidence that can include
historical weights, diagnostics, validation notes, and sensitive development
byproducts.

Directory-wide trust is not allowed. Broad read access would make it easy for a
later tool or agent wrapper to ingest credentials, raw LLM cache files, model
checkpoints, notebook checkpoints, or unreviewed files. File-level allowlisting
keeps packet claims traceable to reviewed repository artifacts.

## 3. Artifact metadata fields

Each approved artifact must include:

- `artifact_id`: stable identifier used by readers, validators, and packet
  citations.
- `path`: repository-relative path to the approved file.
- `artifact_type`: descriptive type label for the artifact.
- `parser`: one of the approved parser labels.
- `role`: short explanation of how the artifact contributes to the packet.
- `allowed_use`: one of the Phase 6 evidence or context uses.
- `canonicality`: evidence status, limited to the approved categories below.
- `research_only`: must be `true`.
- `human_review_required`: must be `true`.
- `caveats`: explicit limitations that should remain visible during packet
  generation and review.

Manifest-level metadata includes `manifest_version`, `phase`, `scope`,
`generated_for`, `research_only`, and `human_review_required`.

## 4. Approved parser labels

The approved parser labels are intentionally simple:

- `csv`
- `parquet`
- `markdown`
- `pdf`
- `json`
- `image_metadata_only`

`image_metadata_only` permits inventory and provenance use for approved figures.
It does not permit computer-vision extraction of chart values or unsupported
claims from image pixels. Material claims should use source tables or documents
when available.

## 5. Canonicality categories

- `canonical`: primary reviewed evidence for Phase 6 summaries.
- `diagnostic`: supporting diagnostics useful for interpretation, stability
  checks, or methodology discussion.
- `historical`: prior experiment artifacts included only for context.
- `auxiliary`: inventory, provenance, or supporting artifacts that should not
  carry material claims alone.

Historical weights, including approved `results/llm/*_weights.parquet` files,
must remain labeled as historical experiment artifacts. They must not be
presented as target allocations or portfolio recommendations.

## 6. Excluded surfaces

The manifest explicitly excludes:

- `.env`
- `.claude/`
- `.venv/`
- `data/cache/llm/*.json`
- `notebooks/.ipynb_checkpoints/`
- `notebooks/_rl_dev/`
- `results/rl/**/agent_seed_*.pt`
- `__pycache__`
- `.pytest_cache`
- arbitrary filesystem paths not listed in the manifest

These exclusions protect secrets, private development state, raw LLM caches,
model weights, generated caches, and unreviewed filesystem content. Exclusion
rules are defensive documentation; enforcement should still rely on
`forbidden_default: true` and the explicit approved artifact list.

## 7. Later deterministic tool usage

Later deterministic tools should use the manifest as their first input. A
reader should load `configs/phase6_artifact_manifest.yaml`, verify
`forbidden_default: true`, validate each requested path against
`approved_artifacts`, and dispatch only to the parser label declared for that
artifact.

Packet generation should record each artifact read, its parser, its
`artifact_id`, and any citation reference used for a material claim. Validators
should fail when a packet cites an unapproved path, omits required provenance,
uses a parser not listed in the manifest, or converts historical diagnostics
into recommendation language.

## 8. Agent framework boundary

Agent frameworks must consume this manifest rather than bypass it. Agno, OpenAI
Agents SDK, or any other wrapper may orchestrate deterministic tools later, but
the framework must not own artifact policy, expand the read surface, connect to
live web or market data, invoke broker APIs, or reinterpret historical weights
as allocation recommendations.

The deterministic manifest is the source of truth for read access. A wrapper
that needs an artifact not listed here should fail closed and require a reviewed
manifest update.

## 9. Phase 6B acceptance criteria

Phase 6B is acceptable when:

- `configs/phase6_artifact_manifest.yaml` exists.
- The manifest has `forbidden_default: true`.
- Manifest metadata includes `manifest_version`, `phase`, `scope`,
  `generated_for`, `research_only`, and `human_review_required`.
- Approved artifacts are listed at file level with the required metadata fields.
- The listed parser labels and allowed uses match the Phase 6 design.
- Model weights, raw LLM cache files, credentials, private development
  directories, and arbitrary directories are not approved.
- `docs/design/phase6b_artifact_manifest_spec.md` documents the purpose,
  allowlist policy, metadata fields, parser labels, canonicality categories,
  excluded surfaces, deterministic tool usage, agent framework boundary, and
  acceptance criteria.
- No agent orchestration, Agno code, OpenAI Agents SDK code, LLM calls, live
  web access, broker access, or market-data refresh capability is implemented.
