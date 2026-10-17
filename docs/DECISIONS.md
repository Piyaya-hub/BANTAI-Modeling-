# Decision register

## Approved direction

Source: the project owner's Phase 0 task instructions. These are approved constraints, not experimental findings.

| ID | Decision | Consequence |
| --- | --- | --- |
| D-001 | Initial repository development scope is trash detection only. | Broader street-issue and hotspot features remain outside initial implementation. |
| D-002 | Evaluate a pretrained Grounding DINO-family model on still images first. | Do not start with training, alternative models, or integration. Exact model selection is unresolved. |
| D-003 | Current work is Phase 0 documentation only. | No detection code, ML dependencies, model downloads, frontend, backend, or Phase 1 execution. |
| D-004 | Use a documentation branch and review against `main`; preserve `.gitignore`. | No direct `main` changes or pull request merge in this task. |
| D-005 | Separate approvals, proposals, unresolved choices, and measured evidence. | Do not invent model versions, benchmark results, or successful training/evaluation claims. |

## Newly confirmed owner decisions — 2026-10-17

Source: the project owner's final Phase 0 revision request. The original D-001–D-005 entries above are retained as history. These approvals establish direction, not successful model behavior or authorization to begin Phase 1.

| ID | Approved decision | Consequence |
| --- | --- | --- |
| D-006 | Repository responsibility is AI modelling and computer vision only. | Future work may include evaluation, preprocessing, detection, necessary postprocessing, approved optimization/fine-tuning, and reusable detection outputs. Frontend/backend application modules, central databases, reporting, incidents, and hotspot ranking are outside this repository. Supersedes any earlier Phase 4 suggestion to implement those modules here. |
| D-007 | Start with pretrained GroundingDINO-T (Swin-T / Tiny configuration). | Refines D-002: the model configuration family is now selected; exact implementation, checkpoint source/revision, configuration, dependencies, and local compatibility remain unresolved for Phase 1. No training from scratch. |
| D-008 | Require Garbage Pile, Individual Litter, and Group of Individual Litter outputs. | Definitions and counting example are in [SCOPE.md](SCOPE.md); operational annotation/grouping/overlap rules remain unresolved. These are not guaranteed pretrained classes. |
| D-009 | Evaluate direct text-guided detection first. | Investigate separate and combined prompts before adding spatial grouping. Illustrative vocabulary is not final. |
| D-010 | Support multiple independent regions and mixed categories in one image. | Preserve objects versus regions and prevent double-counting groups and their members as independent regions; actual direct-model capability remains experimental. |
| D-011 | Grouping is conditional on baseline evidence and explicit owner authorization. | DBSCAN, scale-normalized proximity, custom bounding-box grouping, and hybrid methods remain candidates, not mandatory components. |
| D-012 | Available development GPU reported by owner: NVIDIA RTX 2080 Super. | Verify actual runtime availability, dependency compatibility, and resource needs during Phase 1; no local hardware validation is claimed. |
| D-013 | Final Phase 0 revisions use a new branch and pull request from latest `main`. | PR #1 is merged; use `docs/finalize-phase-0-detection-scope`, do not reuse that PR, merge, or begin Phase 1. Owner review of these revisions remains pending. |

## Proposals awaiting approval

- P-001: Use the staged local inference design in [ARCHITECTURE.md](ARCHITECTURE.md).
- P-002: Use the evaluation protocol outline in [TEST_PLAN.md](TEST_PLAN.md), with concrete choices settled before execution.
- P-003: Use the conditional progression and future gates in [PHASES.md](PHASES.md). Only Phase 0 execution is currently authorized.

## Unresolved technical decisions

| Area | Required decision or evidence | Resolve by |
| --- | --- | --- |
| Model | GroundingDINO-T exact implementation, checkpoint source/revision, configuration, access, provenance, and license | Phase 1 acceptance |
| Runtime | Language/tool versions, dependency pins, local compatibility and RTX 2080 Super availability/suitability, memory and storage budget | Phase 1 acceptance |
| Data | Sources, permissions/privacy, sample selection, three-category operational definitions, annotations, minimum grouping criteria, membership, overlap rules, and ambiguous cases | Phase 1 acceptance |
| Protocol | Prompt vocabulary/formatting/combinations, separate versus combined experiments, thresholds/selection procedure, matching/overlap rule, object-level versus region-level metrics, success criteria, and held-out split | Phase 1 acceptance |
| Outputs | Bounding-box convention, phrase/category mapping, score semantics, object/region provenance and membership, duplicate resolution/counting, record schema, failure representation, artifact retention | Phase 1 acceptance |
| Next steps | Whether baseline evidence supports prompt/threshold changes, spatial grouping, hybrid detection, alternatives, fine-tuning, or integration readiness | Phase 3 review |
| Broader features | Incident identity, location handling, duplicate observations, frequency window, hotspot ranking, and other issue classes | Before separately authorized work; application modules remain outside this repository |

## Recording future decisions

For each approval, record the decision, approving owner, date, rationale, evidence links, and affected scope or phase. A tool suggestion is not an approval. Move a proposal to approved status only after explicit owner approval; preserve superseded decisions with their replacement reference. No training or evaluation results exist in this documentation deliverable.
