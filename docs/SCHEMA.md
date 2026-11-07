# BANTAI Schema Contracts & Unified Output Specifications

> **Scope**: Official JSON data contracts for object detection inferences, spatial hazard regions, diagnostics, and dual-model web application responses.

---

## 1. Executive Summary & Purpose

This document defines the strict, machine-readable JSON schemas produced by the BANTAI computer vision pipeline (Model A: Trash Detector & Spatial Grouping) and consumed by downstream web dashboards, incident management systems, and the municipal Hotspot Prioritization Engine.

All pipeline outputs adhere to:
1. **Schema Versioning**: Versioned contracts (e.g. `bantai.image-detection.v1`, `manifest.json`).
2. **Numeric Safety**: All non-finite floats (`NaN`, `Infinity`, `-Infinity`) or negative bounding box dimensions are serialized as `null` with accompanying error diagnostics.
3. **Disjoint Spatial Regions**: Non-overlapping bounding boxes for multi-item regions.
4. **Independent Lifecycle Status**: Inference status (`ok`, `no_detections`, `error`) is decoupled from display status (`pending`, `shown`, `error`, `disabled`, `not_attempted`).

---

## 2. Single-Image & Video Frame Manifest Schema (`manifest.json`)

Each evaluated media frame produces a structured manifest record adhering to this schema:

```json
{
  "frame_id": 10001,
  "timestamp_sec": 0.0,
  "image_path": "runs/video_detection/incident_001/frames/frame_0001_t00.0s.jpg",
  "width": 1920,
  "height": 1080,
  "summary": {
    "detection_count": 10,
    "region_count": 2,
    "garbage_piles": 1,
    "litter_groups": 0,
    "individual_items": 1
  },
  "detections": [
    {
      "query_id": 14,
      "category_id": 8,
      "label": "Paper",
      "model_score": 0.3542,
      "bbox": [540.2, 632.1, 48.5, 52.0],
      "bantai_category": "Paper"
    }
  ],
  "regions": [
    {
      "region_id": 1,
      "category": "garbage_pile",
      "display_label": "Garbage Pile",
      "item_count": 9,
      "bbox": [480.0, 590.0, 420.0, 310.0],
      "model_score": null,
      "member_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    },
    {
      "region_id": 2,
      "category": "individual_litter",
      "display_label": "Individual Litter",
      "item_count": 1,
      "bbox": [1210.0, 840.0, 35.0, 40.0],
      "model_score": 0.2845,
      "member_indices": [9]
    }
  ],
  "diagnostics": {
    "inference_status": "ok",
    "display_status": "shown",
    "execution_time_sec": 0.245,
    "warnings": []
  }
}
```

---

## 3. Candidate Object Detection Field Specifications

| Field Name | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `query_id` | integer | Grounding DINO query slot index (0..899) | Non-negative integer |
| `category_id` | integer | Supervised class index in `0..8` | Strict lock: 0 to 8 |
| `label` | string | Normalized class label matching the object category | E.g., `"Paper"`, `"plastic_bottle"` |
| `model_score` | float | Calibrated sigmoid confidence score | Range: `[0.0, 1.0]` |
| `bbox` | list of 4 floats | Bounding box coordinates: `[x_min, y_min, width, height]` in pixels | Non-negative, within image bounds |
| `bantai_category`| string | Canonical BANTAI category name | One of: `Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous` |

---

## 4. Spatial Region Grouping Schema (D-112 Standard)

Detections are clustered into municipal hazard regions:

```
+---------------------------------------------------------------------------------------+
|  CARDINALITY CONTRACT (D-112 Standard)                                                |
|                                                                                       |
|  * Individual Litter (individual_litter):                                             |
|      k = 1 isolated bounding box                                                      |
|      model_score = float(score)  (Inherits individual detection confidence)           |
|                                                                                       |
|  * Group of Individual Litter (group_of_individual_litter):                           |
|      2 <= k <= 7 bounding boxes clustered via scale-normalized spatial proximity      |
|      model_score = null  (Aggregated hazard region, not an individual model query)    |
|                                                                                       |
|  * Garbage Pile (garbage_pile):                                                       |
|      k >= 8 bounding boxes in a dense, touching/consolidated heap                     |
|      model_score = null  (High-severity municipal accumulation)                       |
+---------------------------------------------------------------------------------------+
```

| Field Name | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `region_id` | integer | Sequential region identifier within frame | Starts at 1 |
| `category` | string | Internal region category enum | `"individual_litter"`, `"group_of_individual_litter"`, `"garbage_pile"` |
| `display_label` | string | Human-readable region label | `"Individual Litter"`, `"Group of Individual Litter"`, `"Garbage Pile"` |
| `item_count` | integer | Number of detected items in this region ($k$) | $\ge 1$ |
| `bbox` | list of 4 floats | Bounding box enclosing the entire region `[x, y, w, h]` | Non-overlapping with other multi-item regions |
| `model_score` | float or null | Individual confidence for $k=1$; strictly `null` for $k \ge 2$ | Null for multi-item regions |
| `member_indices`| list of integers| 0-based indices into the frame's `detections` array | Contains exactly `item_count` elements |

---

## 5. Diagnostics & Validation Contract

1. **Invalid Numeric Values**:
   Any coordinate or score containing `NaN`, `Inf`, or `-Inf` must be replaced with `null`, excluded from valid summaries and displayed bounding boxes, and flagged in `diagnostics.warnings`.
2. **Unmapped Phrases**:
   Predictions failing exact text-label mapping retain their raw decoded phrase string with `category_id = null`.
3. **Display Status Lifecycle**:
   - `pending`: Inference persisted to disk; display window not yet opened.
   - `shown`: GUI window opened and normally closed by user.
   - `error`: GUI display failed (e.g. headless environment, missing TkAgg driver).
   - `disabled`: Display suppressed via CLI flag (`--no-gui`).
   - `not_attempted`: Processing failed prior to display invocation.

---

## 6. Unified Dual-Model Response Schema (Model A + Model B)

To power the website's municipal dispatch queue and Leaflet incident map, Argon combines Model A (Trash) and Model B (Road Defects) outputs:

```json
{
  "incident_id": "INC-20261107-0042",
  "source_media": {
    "filename": "patrol_route_zone4.mp4",
    "type": "video",
    "fps": 1.0,
    "duration_sec": 12.0
  },
  "location": {
    "latitude": 14.653889,
    "longitude": 121.103611,
    "street_name": "Rizal Avenue cor. Aurora Blvd"
  },
  "hotspot_score": 88.5,
  "priority_level": "CRITICAL",
  "dispatch_recommendation": "Immediate municipal cleanup crew (waste pile) + road patch crew (severe pothole)",
  "frames": [
    {
      "frame_index": 3,
      "timestamp_sec": 3.0,
      "frame_url": "/media/incidents/INC-0042/frames/frame_0003.jpg",
      "annotated_url": "/media/incidents/INC-0042/overlays/frame_0003_overlay.jpg",
      
      "trash_findings": {
        "model_version": "BANTAI-Stage5-D095-Soup5050",
        "total_items": 12,
        "regions": {
          "garbage_piles": 1,
          "groups_of_litter": 0,
          "individual_items": 2
        },
        "items_by_bantai_category": {
          "Plastic": 7,
          "Paper": 3,
          "Metal": 1,
          "Potentially Hazardous": 1,
          "Organic": 0
        },
        "region_list": [
          {
            "region_id": 1,
            "type": "garbage_pile",
            "item_count": 10,
            "bbox": [410.5, 520.0, 380.0, 290.0]
          }
        ]
      },

      "road_defect_findings": {
        "model_version": "BANTAI-ModelB-RoadDefect-v2",
        "defects": [
          {
            "defect_id": 1,
            "type": "pothole",
            "severity": "high",
            "estimated_diameter_cm": 45.0,
            "bbox": [890.0, 720.0, 160.0, 110.0],
            "confidence": 0.892
          }
        ]
      }
    }
  ]
}
```

For complete integration instructions, see [Master Integration Guide](ARGON_WEBSITE_INTEGRATION_GUIDE.md).
