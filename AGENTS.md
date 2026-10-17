# BANTAI agent instructions

## Project boundary

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. The initial development scope of this repository is **trash detection on still images only**. Roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization belong to the broader proposal; they are not all approved implementation work here.

Phase 0 establishes documentation only. Do not start Phase 1 without explicit authorization. Do not implement detection code, install machine-learning dependencies, download models, or create a frontend or backend during Phase 0.

## Collaboration rules

- ChatGPT supports requirements analysis, explanations, and proposed decisions. Label suggestions as proposals; do not present them as approvals or measured results.
- Codex inspects the repository, makes authorized changes on a task branch, checks consistency, and reports actual commands and outcomes. Preserve existing files and user changes, including `.gitignore`; never commit credentials or unrelated artifacts.
- Antigravity follows the same scope and evidence requirements when inspecting, proposing, implementing, or reviewing authorized work. Its suggestions do not authorize scope expansion.
- All three tools must read this file and the relevant documents before work, distinguish facts from assumptions, and record unresolved choices rather than inventing answers.
- The project owner approves scope changes, phase transitions, and decisions. Tools may propose changes through a pull request but must not claim owner approval or merge without authorization.
- Use the existing isolated checkout; do not create a Git worktree unless requested. Do not change `main` directly. Use focused branches and pull requests targeting `main`.
- Do not claim that a model has been trained, evaluated successfully, or benchmarked without recorded evidence. Do not invent exact model versions, datasets, or benchmark results.

## Documentation maintenance

Keep approved decisions in [docs/DECISIONS.md](docs/DECISIONS.md) separate from proposals and unresolved questions. Update [docs/HANDOFF.md](docs/HANDOFF.md) with the actual state, evidence, blockers, and next authorized action. Check relative Markdown links and phase acceptance criteria before committing.

Start with [README.md](README.md), [scope](docs/SCOPE.md), and [phases](docs/PHASES.md). For future authorized model work, follow the [inference architecture](docs/ARCHITECTURE.md) and [test plan](docs/TEST_PLAN.md).
