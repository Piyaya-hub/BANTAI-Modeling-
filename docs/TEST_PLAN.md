# Software and functional detection checks

Scrollable-tab amendment: test exact repeated section order, raw/incomplete phrases, bounded readable labels above narrow/edge boxes, coordinate alignment, passing-only selectable tables, actual saved configuration, wheel/horizontal scrolling, multi-image replay, normal closure and batch failures. Keep synthetic/Agg/mock evidence separate from actual Windows visual observations. See [current GUI evidence](SCROLLABLE_VIEWER.md).

## Current scope and evidence

Phase 2A readiness and Phase 2B foundation are published. The owner has authorized Phase 2D display completion and validation; see [2D checks](PHASE_2D_CHECKS.md). Phase 2C baseline inference/mapping remains unchanged. Software checks now include loader/device/API, exact mapping/duplicate/overlap behavior, persistence ordering and viewer failure handling. Actual commands/results are in [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). All fake predictions remain synthetic software fixtures; Agg canvas drawing is not native rendering, and no actual cloud model run occurred.

The owner manually downloads/extracts unannotated team photographs from Google Drive into a project folder. Current model checking is functional visual inspection, not formal ground-truth evaluation. No annotation work, benchmark split, precision/recall/mAP, or numerical accuracy target is required. Software fixtures verify program behavior and are not ground-truth evaluation of the model.

## Documentation checks

Check all tracked Markdown references, parse every fenced JSON example strictly, and review current-state/category/phase/output/status consistency. Confirm existing ignore rules, including `/data/input/`, `/results/`, and `/.cache/huggingface/`, remain intact. Inspect for already tracked photos/results/caches without silently untracking them. Ensure the Phase 2C diff contains only the authorized baseline, supporting tests and documentation; preserve requirements pins/source and prior evidence. Preserve existing ignore rules and append `/work/` for synthetic/scratch artifacts; no tracked images, results, caches or environments.

## Planned software checks

| Area | Expected behavior to check with fixtures or controlled inputs |
| --- | --- |
| Input handling | Configurable project-relative paths; missing/empty folders actionable; unsupported skips distinct from unreadable-image errors; batch continues and reports errors honestly. |
| Orientation and coordinates | EXIF orientation is applied in memory; original-display dimensions and xyxy pixel-edge convention match viewer and JSON; finite raw values preserved; valid clipped boxes stay inside bounds; invalid geometry never drawn. |
| Invalid numbers | Each nonfinite raw coordinate becomes null with indexed diagnostics; any such coordinate makes the whole display box null. Invalid score becomes null with a reason. Invalid candidates are not displayed or counted as valid detections. Preserve other finite raw components; never emit NaN/Infinity. An all-invalid candidate result is diagnosed, not `no_detections`. See the [numeric contract](PHASE_1_IMPLEMENTATION_PLAN.md#invalid-numeric-values-and-candidate-validity). |
| Mapping | Exact normalized aliases only; partial/combined/unmatched phrases remain null-category, visible as unmapped, with raw phrase and valid model score retained. Mapping does not prove visual category identity. |
| Duplicates and overlaps | Identical-box/same-phrase proposal collapses display/summary only and retains duplicate records/links. Near duplicates remain; IoU/containment flags do not implement clustering, membership, or general NMS. Conflicting categories remain inspectable. Unresolved physical-object/independent-region counts stay null. |
| Processing status | Distinguish `ok`, `no_detections`, `error`, and `skipped_unsupported`. `no_detections` requires successful inference with zero above-threshold candidates; empty valid summaries after invalid predictions are not proof of no trash. Processing errors and display errors have separate fields. |
| Display lifecycle | Atomically persist inference result first with `display.status=pending` when viewing is planned, then attempt display, then atomically update `shown` or `error`. Simulated interruption leaves pending as unfinished. Intentional disablement records `disabled`; processing failures/skips use `not_attempted`. Display failure preserves successful detections/status and records `display.error`. Default viewing is enabled; viewer waiting is excluded from inference timing. |
| Output consistency | Boxes, categories/unmapped labels and individual model-score percentages agree with JSON; percentages are score ×100, never image accuracy. Readable summaries exclude invalid/duplicate candidates as specified; JSON and appropriate run/environment metadata are standards compliant. |
| Source and export protection | Original byte hashes unchanged; no source overwrites, automatic annotated copies, or viewer temp-image files. Default output is JSON/metadata only; optional image export is off. No real photos, results, caches, or secrets enter Git. |

These checks can validate output boundaries without team photos or model downloads. A fake prediction fixture is explicitly software test data, never an observed model result.

## Planned visual functional checks

When photos exist and inference is authorized, inspect `individual_litter`, `group_of_individual_litter`, and `garbage_pile`, mixed-category scenes, and images with no relevant trash. Include, where available:

- Isolated litter far from a group and multiple separate groups in one image.
- Small/distant, overlapping, and partially occluded objects.
- A pile near individual litter, or a pile and scattered group together.
- Missed visible trash, plausible false detections, wrong categories, unmapped phrases, duplicate boxes, missed group-region boxes, and pile/group confusion.

Compare separate and combined prompt runs on unchanged photos with settings and run IDs recorded. No held-out split is required. Keep direct model object predictions separate from region predictions, do not invent memberships/counts, and retain uncertain visual judgments as qualitative notes. The five-object/two-region example in [SCOPE.md](SCOPE.md) is illustrative desired behavior, not a result or guarantee.

Check that native in-memory viewing works with boxes, labels, and individual model-score percentages; match them against per-image JSON. A failed viewer does not erase inference results but leaves the display requirement unmet. A deliberately disabled viewer is not evidence that display works. Check no-detection and malformed/unsupported input paths without claiming those fixtures test detection quality.

Record model load time separately; timed observations specify device/GPU/driver/packages, preprocessing/image dimensions, warm-up and repetitions, precision and timing scope. Synchronize GPU timing and exclude human viewer waiting. No score or timing observation establishes precision, recall, mAP, or accuracy.

## Reporting and future scope

Keep passed, failed, skipped, and unrun outcomes separate. Record actual commands/settings, fixtures versus real-photo observations, artifact provenance, resource/timing scope, and disclosed limitations. Mark unavailable visual scenarios unrun. A successful import, zero-test run, published plan, or open viewer does not demonstrate successful detection.

If direct prompting cannot distinguish groups and piles or localize group regions, report the limitation. Grouping, training, fine-tuning, or alternative-model work requires a later evidence-based decision and authorization. Formal annotated evaluation may be separately scoped in the future; its annotations, benchmark splits and accuracy metrics are not current prerequisites.

See [PHASES.md](PHASES.md) for phase gates and [HANDOFF.md](HANDOFF.md) for actual state and remaining verification work.
