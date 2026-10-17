# Phase 2B — software checks and evidence

Implemented on 2026-10-17, branch `feat/phase-2b-io-foundation`, from published main `dde98cf0642672eb96422891e101c21c6b67bbc2` (merge of PR #5). Initial tracked checkout was clean; existing branches, ignored environment and user files were preserved. No worktree or direct-main changes. Shell Git fetch failed because the configured proxy could not be reached. The connected GitHub tools supplied current-main metadata and file contents; blob/tree/commit object hashes were verified locally before switching to the exact base. The base is locally shallow because its new parent objects were unavailable through shell Git. This does not change the published base or proposed diff.

## Actual verification

Commands executed in the cloud checkout:

| Command/check | Actual result | Scope |
| --- | --- | --- |
| `/usr/bin/python3 -m unittest discover -s tests -v` | 22 tests passed, zero failures/errors | Synthetic software checks only |
| `/usr/bin/python3 scripts/detect_images.py --help` | Exit 0; explicitly says preparation only, no inference or viewer | CLI entry point |
| `/usr/bin/python3 -m compileall -q scripts tests` | Exit 0 | Syntax/import compilation only |
| `git diff --check` | Exit 0 | Whitespace check |
| Documentation/snapshot consistency script | 84 local Markdown links valid; 5 fenced JSON examples parse strictly; all 36 requirements pins equal the supplied snapshot; original ignore rules preserved | Read-only checks, no dependency resolution/install |

The test runner is existing cloud Python **3.13.5**, Pillow **11.1.0**. The cloud `.venv` lacks Pillow; the initial import probe reported `ModuleNotFoundError`. No package was installed or reinstalled. The existing system Pillow was used for software checks. These are not executions on the verified Windows Python 3.11.9/Pillow 11.2.1 environment and do not replace its supplied 2A evidence. Running the new software tests on that unchanged target remains unrun; no fresh-install, Windows filesystem or GPU result is claimed by 2B.

Fixtures exercise:

- Repository-root path resolution from another working directory; absolute overrides; missing/invalid/duplicate-key/nonstandard JSON configuration; threshold type/range/nonfinite checks; pinned model/processor and device/display validation; path overlaps including symlinks.
- Sorted nonrecursive discovery; missing/empty/not-directory inputs; unsupported skips; corrupt images, wrong file content, missing/unreadable sources; continuation after per-file failures.
- Pixel-exact synthetic PNGs for all eight EXIF orientations; oriented dimensions and box clipping; RGB conversion from grayscale, RGBA and CMYK JPEG; byte SHA-256 equality before/after preparation; no generated image copies.
- NaN/Infinity/non-numeric coordinates and scores; reversed/degenerate/off-image boxes; finite raw values and out-of-range score diagnostics; valid clipping; invalid candidates excluded from valid summaries; unmapped phrases retained; unresolved physical/independent-region counts.
- `ok`, `no_detections`, all-invalid `error`, pre-inference errors and unsupported outcomes; inconsistent records rejected; explicit inference-completion/provenance guards. Every prediction is synthetic and appears only in tests. No model supplies predictions.
- Pending/disabled/not-attempted and simulated shown/display-error outcomes; interrupted pending record retained; final update failure leaves earlier bytes unchanged. These call persistence helpers only, with no implemented viewer.
- Distinct UUID run reservations; forced UUID collision refusal; full path/content source identities; no basename collisions or initial overwrites; failed initial/final publication; simulated write/flush/fsync/replace errors; strict serialization failures and temporary-file cleanup.
- Preparation CLI mixed batch and unsupported-only/missing/empty cases; JSON-only outputs, explicit inference-unrun flag, and no functional success/no-detections after decoding.

All temporary files, source fixtures and JSON artifacts live under ignored `work/` and are cleaned by the tests. No team photos, model cache, weights, image copies or generated results are tracked. Only JSON is persisted by the CLI, under the configured ignored results path by default.

## Implementation and limits

`config/detection.json` preserves the verified model/processor revision, with prompts/thresholds explicitly proposed and unvalidated. Display defaults to enabled, but preparation never invokes it. Paths are resolved before output; input/results/cache paths cannot overlap. The result helpers support Section 7 candidate sanitation, consistency checks, strict persistence and display-only finalization; they do not implement phrase mapping, duplicate selection, overlap heuristics, detector extraction or a viewer.

Functional results add `inference.completed` and require actual device plus environment/preprocessing manifest references for completed inference; the later integration must provide observed provenance and verify its installed library. No software guard can independently prove a caller actually ran a model. The preparation CLI uses separate schemas, `bantai.preparation.v1` and `bantai.preparation-run.v1`; its `prepared` state cannot be mistaken for functional inference success. Unsupported source bytes are not decoded or hashed; unreadable sources may also have null hashes. Successfully read supported sources, including decode failures, retain the actual SHA-256.

Run directories are reserved exclusively with UUIDs. Source output names digest full source path identity plus byte digest; hash-based identity follows SHA-256 collision resistance, while exclusive publication also refuses an existing destination. Project inputs retain project-relative IDs; explicitly external inputs use resolved absolute IDs. Initial publication requires hard-link support and fails observably otherwise; explicit updates use same-directory `os.replace` after flush/fsync/close. One writer per result is required. Atomic visibility and previous-file preservation are tested; directory power-loss durability, network filesystems and Windows behavior were not tested here. A failed run leaves its existing manifest `in_progress`, identifying unfinished work, and returns an observable error; it is not silently marked complete.

Requirements files transcribe all 36 supplied pins from [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md), including bootstrap tools, and retain `https://download.pytorch.org/whl/cu118`. They are version snapshots without artifact hashes, not evidence of a tested fresh installation. No dependency changes or new packages were introduced.

## Readiness for 2C

The bounded foundation is ready for review and for later integration when the owner authorizes 2C. Before treating the integrated pipeline as target-validated, run these new software tests in the existing Windows environment. Preserve the recorded model/processor revision and 2A package/source evidence. Model loading/memory fit, real inference, custom-category quality, mapping/duplicates/overlap behavior and native photo overlays remain unrun. No team photos or GPU access were required for these software checks. No automatic grouping, training, web/application modules, merge or 2C implementation occurred.
