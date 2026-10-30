# Stage 5 local academic-use record — 2026-10-26

This records the owner's declaration and the documented Section 185 fair-use basis for the Stage 5 9-class model workflow incorporating `paper` (D-070, Option A). The exact corpus comprises:
- **Training set**: 221 frozen original + 86 newly eligible paper images = **307 images** (120 paper boxes + 106 original-class boxes)
- **Validation set / annex**: 47 frozen original + 18 newly eligible paper images = **65 images** (20 paper boxes)
- **Held-out test set / annex**: 47 frozen original + 18 newly eligible paper images = **65 images** (25 paper boxes, strictly held out and uninspected).

## Owner declaration

The owner declares:
1. BANTAI is a non-commercial, academic capstone research project at the Technological Institute of the Philippines (TIP) Quezon City ("An Intelligent Agent for Street Issue Detection and Hotspot Prioritization").
2. All training, evaluation, and visual checks take place locally on the owner's personal computer.
3. No TACO photographs or annotations will be published, redistributed, or uploaded to public cloud infrastructure.
4. The owner authorizes the local fine-tuning and evaluation workflow under Option A.
This record documents the factual basis under Section 185 of Republic Act No. 8293; no formal institutional clearance, court order, or rightsholder waiver is claimed.

## Statutory basis & four-factor analysis

Under the Intellectual Property Code of the Philippines (RA 8293 Section 185, as amended by RA 10372 Section 12) and IPOPHL statutory fair-use guidelines:

| Factor | Assessment for Stage 5 expanded corpus |
| --- | --- |
| **Purpose & character** | Non-commercial academic object detection research. The workflow performs statistical parameter updates on a vision-language model, not image redistribution or public display. Strongly favors the claim. |
| **Nature of work** | Street and environmental litter scenes photographed in context. Informational and factual in nature, though original composition remains protected expression. Mixed/uncertain. |
| **Amount & substantiality** | Whole images are analyzed strictly to retain spatial context for bounding boxes. Only 307 training and 65 validation images are processed locally out of the 1,500 TACO corpus. Whole-photo processing is technically necessary for object localization. |
| **Effect on potential market** | Zero market substitution: no commercial product, licensing service, or public image repository is created. The research does not impair Flickr or rightsholder commercialization. |

## Machine-readable record & governance

The authorization uses `bantai.training-rights.v2`:
- `workflow_authorization`: `true`
- `photo_clearance`: `false` (fair-use exception claimed, not licensed)
- `database_clearance`: `false` (fair-use exception claimed)
- `definitive_legal_clearance`: `not_claimed`
- Bound corpus: 307 training, 65 validation, and 65 held-out test images.


## D-074: versioned Tier 1 corpus authorization ? 2026-10-29

The owner explicitly authorizes a local non-commercial academic Tier 1 continuation
under the same documented Section 185 claim, with **336 training images / 637 boxes**
and the unchanged **65 validation images / 109 boxes**. This is 401 unique authorized
train/validation photographs; the 65 test images remain frozen and uninspected.
The new corpus incorporates 30 downloaded strict-clean images and excludes Image 1173
by exact-duplicate propagation from Image 1305 with Unlabeled litter.

The versioned export is `exports/taco/strict-clean-stage5-tier1-v1`; its new record is
`work/stage5-paper/rights_tier1.json`. That record binds exact export/annotation/taxonomy
hashes and the 401 verified source-photo SHA-256 digests. It authorizes one 336-update
384/576 FP32 epoch at LR 2e-6 and three 112-forward validation evaluations (epoch state
and two in-memory soups), with no test use or automatic promotion/publication.

Purpose remains local academic localization research. Nature remains protected photos
and potentially protected annotation/database expression. Whole-image amount now
covers 336 train and 65 validation images, not the historical 307/65 corpus; necessity
for spatial context does not remove uncertainty. Market effects remain unmeasured;
no redistribution, uploads, commercial clearance or definitive legal conclusion is
claimed. Owner workflow authorization remains distinct from rightsholder permission,
institutional approval and definitive legal clearance; all three remain not claimed.

Historical D-070 record and its 307/65 corpus remain unchanged. The exact earlier
assessment bytes are archived by SHA-256 under `docs/assessments/`; the Stage 5 rights
gate accepts only an archive matching the hash already bound by the old record.
This preserves that record without pretending its authorization covers the new corpus.
