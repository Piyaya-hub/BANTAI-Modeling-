# Development phases

Every transition requires explicit project-owner authorization. Acceptance criteria define evidence to review; listing a phase does not authorize it. Later phases are conditional and may be revised or declined.

| Phase | Work | Acceptance criteria |
| --- | --- | --- |
| 0 — Foundation documentation | Document scope, inference design, decisions, validation, and tool collaboration rules. | All eight required Markdown files exist; local references resolve; initial trash-only scope is consistent; approved decisions and proposals are separate; unknowns are explicit; no model code, ML dependencies, or downloaded models are added; `.gitignore` is preserved; documentation is committed on a focused branch and offered for review. |
| 1 — Evaluation preparation | After authorization, choose a pretrained Grounding DINO-family implementation and checkpoint; determine runtime and device; define image and annotation policy and evaluation protocol. | Selection, provenance, license, environment requirements, input/output contract, dataset rights, annotation rules, prompts, threshold-selection procedure, metrics, and owner-approved success criteria are recorded. Any required environment setup is verified. No unresolved prerequisite needed for evaluation remains. |
| 2 — Still-image evaluation | After authorization, implement and run the approved pretrained-model experiment on trash images. | Actual execution evidence is retained; input failures, empty outputs, and box coordinates are checked; annotated evaluation reports metrics under the recorded protocol, representative failures, resource observations, and limitations. Reproduction instructions work. Report unmet success criteria honestly; phase review may recommend revision rather than progression. |
| 3 — Evidence-based next-step decision | Review evaluation findings before selecting further work. | Owner records whether to retain the baseline, revise evaluation, investigate alternatives, consider training, or stop. Any approved experiment has its own scope, acceptance criteria, resource/data requirements, and authorization before execution. |
| 4 — Conditional integration or broader research | Only separately approved work, potentially integration, incident workflows, other issue classes, or frequency-based hotspots. | Before implementation, each selected capability has an approved design, data policy, measurable acceptance criteria, and validation plan. Completion requires recorded evidence against those capability-specific criteria and owner review; none is claimed complete now. |

## Present gate

Only Phase 0 is authorized by the current task. Phases 1–4 are not started. Meeting Phase 0 documentation checks is not evidence that inference works or that evaluation passed. See [HANDOFF.md](HANDOFF.md) for current state and [DECISIONS.md](DECISIONS.md) for decision status.
