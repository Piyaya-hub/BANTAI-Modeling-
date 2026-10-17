# Phase 2A verified environment snapshot

## Status and provenance

Phase 2A technical preparation checks are complete on the Windows target based on the owner's supplied command outputs, screenshot and explicit normal-viewer-return confirmation. Codex guided the commands from a separate Linux cloud checkout and recorded the evidence; it did not directly execute Windows commands. This records preparation readiness, not owner acceptance of later implementation, model-memory fit, successful detection or accuracy.

The owner selected a local `Documents/Python/Project Image-object Detection` repository folder under their user profile. Its fresh `.venv` uses Python 3.11.9 x64. No prior user environment was replaced. The initial RTX 2080 Super report was superseded by actual supplied GTX 1660 Ti output; historical decision rows remain unchanged.

## Recorded target and checkpoint metadata

The following values are actual supplied observations, not fictional detection examples. Free VRAM is a past query observation, not guaranteed future availability. The model/config/API/card/preprocessor requests downloaded documentation/configuration only, not model weights. Both future model and processor loads must use the recorded revision.

```json
{
  "evidence_source": "Owner-supplied Windows command outputs, native-plot screenshot and explicit normal-return confirmation; not cloud-executed Windows checks",
  "phase": "2A environment preparation only",
  "os_version": "10.0.19045.6466",
  "git_version": "2.45.1.windows.1",
  "python_version": "3.11.9",
  "python_bits": 64,
  "environment_relative_path": ".venv",
  "device": {
    "name": "NVIDIA GeForce GTX 1660 Ti",
    "driver": "472.19",
    "total_memory_mib": 6144,
    "observed_free_memory_mib": [
      5235,
      5236
    ],
    "compute_capability": [
      7,
      5
    ],
    "cuda_build": "11.8",
    "cuda_available": true,
    "cpu_tensor_result": 2.0,
    "gpu_tensor_result": 2.0
  },
  "dependency_check": "No broken requirements found.",
  "native_viewer": {
    "backend": "TkAgg",
    "synthetic_plot_rendered": true,
    "normal_return_confirmed": true,
    "annotated_image_export": false
  },
  "model": {
    "id": "IDEA-Research/grounding-dino-tiny",
    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "license_metadata": "apache-2.0",
    "gated": false,
    "safetensors_files": [
      "model.safetensors"
    ],
    "observed_model_type": "grounding-dino",
    "observed_backbone": {
      "depths": [
        2,
        2,
        6,
        2
      ],
      "model_type": "swin",
      "num_heads": [
        3,
        6,
        12,
        24
      ],
      "out_features": [
        "stage2",
        "stage3",
        "stage4"
      ],
      "out_indices": [
        2,
        3,
        4
      ]
    }
  },
  "processor": {
    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "config": {
      "do_normalize": true,
      "do_pad": true,
      "do_rescale": true,
      "do_resize": true,
      "format": "coco_detection",
      "image_mean": [
        0.485,
        0.456,
        0.406
      ],
      "image_processor_type": "GroundingDinoImageProcessor",
      "image_std": [
        0.229,
        0.224,
        0.225
      ],
      "processor_class": "GroundingDinoProcessor",
      "resample": 2,
      "rescale_factor": 0.00392156862745098,
      "size": {
        "longest_edge": 1333,
        "shortest_edge": 800
      }
    }
  },
  "cuda_install_index": "https://download.pytorch.org/whl/cu118",
  "supporting_install_command": "python -m pip install transformers==4.51.3 Pillow==11.2.1 matplotlib==3.10.3",
  "local_artifacts": [
    ".venv/phase2a-requirements.txt",
    ".venv/phase2a-environment.json"
  ],
  "model_weights_downloaded_in_phase2a": false,
  "model_loaded_in_phase2a": false,
  "inference_run_in_phase2a": false
}
```

The processor configuration's `coco_detection` format is an upstream processor setting, not a requirement to annotate team photos. The baseline remains unannotated functional checking in a later authorized phase.

## Official sources and driver interpretation

- [Pinned model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md): the supplied excerpt describes zero-shot detection and lower-case period-separated prompts; it does not validate BANTAI's categories.
- [Pinned model configuration](https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/config.json) and [processor configuration](https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/preprocessor_config.json): target HTTP responses were successful; metadata values are recorded above.
- [NVIDIA CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html): supplied target output from Table 2 lists Windows x86_64 driver >=452.39 for CUDA 11.8.x minor-version compatibility. Reported driver 472.19 exceeds that minimum, and actual CPU/CUDA tensor checks passed. The separately listed Windows driver 522.06 is associated with the toolkit distribution; it is not the same table's minor-compatibility minimum.
- [Official cu118 wheel index](https://download.pytorch.org/whl/cu118/torch/): cloud metadata inspection confirmed the Windows CPython 3.11 torch 2.7.1+cu118 wheel; target installation/import/device checks supplied the runtime evidence.

The earlier cloud proxy 403 responses remain historical access observations. Successful target HTTP requests resolved the checkpoint and driver-reference verification gaps; they do not imply that cloud network policy changed. Minor compatibility and a basic tensor test do not guarantee every future model operation or memory requirement. No driver/toolkit update was performed or required for the checks completed here; diagnose any future authorized model-specific failure using actual evidence.

The API reported Apache-2.0 license metadata and non-gated access. Preserve upstream license/notice conditions in later authorized distribution; the supplied model-card excerpt showed no additional application restriction. No weights were obtained during preparation.

## Exact dependency snapshot

The owner ran `pip freeze --all` in the target `.venv`. The complete supplied 36-package output is retained below. `pip check` passed and direct imports reported torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3. Pillow's initial transitive version 12.3.0 was explicitly replaced by the planned 11.2.1 before final verification.

```text
certifi==2026.7.22
charset-normalizer==3.5.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
filelock==3.32.3
fonttools==4.67.0
fsspec==2026.7.0
huggingface_hub==0.36.2
idna==3.20
Jinja2==3.1.6
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.10.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.3
pillow==11.2.1
pip==24.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
safetensors==0.8.0
setuptools==65.5.0
six==1.17.0
sympy==1.14.0
tokenizers==0.21.4
torch==2.7.1+cu118
torchvision==0.22.1+cu118
tqdm==4.70.1
transformers==4.51.3
typing_extensions==4.16.0
urllib3==2.8.0

```

CUDA torch/torchvision were installed with `--index-url https://download.pytorch.org/whl/cu118`; supporting packages used the command recorded above. The target saved `.venv/phase2a-requirements.txt` and `.venv/phase2a-environment.json`, confirmed by command output. The JSON contains Python/checkpoint/processor settings and the CUDA installation source; the full freeze includes bootstrap tools. These local files remain ignored under `.venv`. This committed document preserves their supplied package/configuration values without committing an environment, photographs, caches or weights.

This is a point-in-time Windows environment snapshot, not an artifact-hash lock or a claim that a fresh reinstall was tested. Keep the CUDA install source alongside versions when reproducing; a bare local-version freeze is insufficient to select the cu118 wheel source. Do not rerun passed readiness checks on the unchanged environment without a new failure or concern. If rebuilding or changing packages, revalidate the affected package/device/viewer operations. Python 3.11.9 came from the official python.org x64 installer through WinGet, whose supplied output confirmed installer-hash verification; no TLS/checksum checks were disabled.

## Completed and deliberately unrun work

Passed: interpreter/venv setup, exact direct package imports, dependency consistency, CPU/CUDA tensor execution, official checkpoint/configuration/license/safetensors metadata, same-revision processor settings, native TkAgg rendering/normal return, exact freeze and install-source capture. No current Phase 2A preparation blocker remains in the supplied evidence.

Not run by design: pretrained weight download/loading, detector implementation, inference, team-photo processing, grouping, training/fine-tuning, model-memory-fit tests and detection-quality evaluation. Preparation completion does not authorize them, merge the branch, accept every proposed prompt/threshold/alias rule or approve a later phase. Stop here pending the owner's next phase instruction. See [PHASES.md](PHASES.md), [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md), and [HANDOFF.md](HANDOFF.md).
