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

## Proposals awaiting approval

- P-001: Use the staged local inference design in [ARCHITECTURE.md](ARCHITECTURE.md).
- P-002: Use the evaluation protocol outline in [TEST_PLAN.md](TEST_PLAN.md), with concrete choices settled before execution.
- P-003: Use the conditional progression and future gates in [PHASES.md](PHASES.md). Only Phase 0 execution is currently authorized.

## Unresolved technical decisions

| Area | Required decision or evidence | Resolve by |
| --- | --- | --- |
| Model | Exact implementation, checkpoint/revision, access, provenance, and license | Phase 1 acceptance |
| Runtime | Language/tool versions, dependency pins, device, memory and storage budget | Phase 1 acceptance |
| Data | Sources, permissions/privacy, sample selection, annotations, trash definition and ambiguous cases | Phase 1 acceptance |
| Protocol | Prompts, thresholds and their selection procedure, matching/overlap rule, metrics, success criteria and evaluation split | Phase 1 acceptance |
| Outputs | Bounding-box convention, score semantics, record schema, failure representation, artifact retention | Phase 1 acceptance |
| Next steps | Whether evidence supports alternatives, training, or integration | Phase 3 review |
| Broader features | Incident identity, location handling, duplicate observations, frequency window, hotspot ranking, and other issue classes | Before any separately authorized broader work |

## Recording future decisions

For each approval, record the decision, approving owner, date, rationale, evidence links, and affected scope or phase. A tool suggestion is not an approval. Move a proposal to approved status only after explicit owner approval; preserve superseded decisions with their replacement reference. No training or evaluation results exist in this documentation deliverable.
