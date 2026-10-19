# Stage 4 local academic-use record — 2026-10-19

This records the owner's latest declaration and the documented basis for the existing Stage 4 run; it is not a new preparation phase. The exact corpus is the unchanged strict-clean v1 **221 training / 397 objects and 47 validation / 75 objects**. The local machine-readable record binds all 268 image IDs and split/source hashes. Test photos remain unused.

## Actual owner declaration

The owner states that BANTAI is non-commercial and academic; training/evaluation occur locally on the owner's computer; TACO photographs/annotations will not be published, redistributed or uploaded to cloud services; and the owner authorizes the local fine-tuning workflow. The owner explicitly requested a documented Section 185 analysis and execution if supportable. This chat declaration is the evidence; no signature, rightsholder permission, institutional approval or lawyer's opinion was supplied or is claimed.

## Sources and four-factor application

The [TACO maintainer README](https://github.com/pedropro/TACO/blob/master/README.md) expressly describes the images/annotations as supporting training and evaluation of detection algorithms. The [authors' paper](https://arxiv.org/abs/2003.06975) describes the litter dataset and segmentation experiments. These support research-purpose compatibility, not blanket photo licensing. Retain citation: Pedro F. Proença and Pedro Simões, *TACO: Trash Annotations in Context for Litter Detection*, 2020, arXiv:2003.06975.

[RA 8293 Section 185](https://lawphil.net/statutes/repacts/ra1997/ra_8293_1997.html), with [RA 10372 Section 12's amendment](https://lawphil.net/statutes/repacts/ra2013/ra_10372_2013.html), provides the research/fair-use framework. [IPOPHL's guidance](https://www.ipophil.gov.ph/news/ipophl-releases-statutory-fair-use-guidelines-to-clarify-rules-on-copyright-exceptions/?news_page=2) calls for considering all four factors; non-commercial status alone is insufficient. The following is an evidence-based project assessment, not a definitive legal determination:

| Factor | Application to this specific use |
| --- | --- |
| Purpose/character | Non-commercial local academic object-localization experiments, consistent with TACO's documented purpose. Analytical model fitting/evaluation differs from publishing photographs. Favors the claim; research is not automatic immunity. |
| Nature | Litter scenes are informational, but photographic composition and annotation selection can be protected expression. Factual coordinates alone do not exhaust database/compilation interests. Mixed/uncertain; public availability is not public domain. |
| Amount/substantiality | Whole images and associated annotations are consumed to preserve spatial context and valid boxes. Only the established train/validation subset is used, for a bounded two-epoch experiment. Entire-photo copying weighs against the claim despite technical necessity; a small fraction of TACO alone is not decisive. |
| Market/value | No image/annotation distribution, commercial service or public substitute corpus is created. This limits substitution risk. No market study or per-image licensing-market evidence was supplied; potential licensing and model-related markets remain uncertain. |

For annotations/database interests, the same limited analytical use is assessed to the extent protectable selection/arrangement is reproduced. [Sections 173 and 175](https://lawphil.net/statutes/repacts/ra1997/ra_8293_1997.html) distinguish original compilations from mere data; neither makes all annotations automatically unrestricted. Preserve original metadata, credits and ODbL notices. No contractual waiver, foreign-law exemption or resolution of every individual photograph's license is claimed. Existing missing/generic-CC/ODbL strings remain unchanged.

## Decision and scope

On the declared facts, the research-purpose evidence, necessity of spatial context and strict local non-substitutive use provide a **defensible, limited Section 185 fair-use claim**, while the whole-image and market uncertainties remain. The owner authorizes this workflow; this record does not purport to grant rights or certify legal clearance. A conclusive court/institutional determination is not asserted. The workflow may proceed on this documented exception claim without inventing a positive license ledger.

The machine record uses `bantai.training-rights.v2`: workflow authorization is true; photo/database clearance flags remain false; rightsholder permission, institutional approval and definitive clearance are explicitly `not_claimed`. The assessment digest and genuine sources are mandatory. Legacy v1 permission/clearance records retain their existing semantics. This is an auditable distinction, not a disabled gate.

Authorized: original-coordinate preflight, pinned-model baseline, measured GPU pilot, two epochs and same-protocol validation on the exact bound train/validation corpus, entirely local. Preserve originals, notices, inference settings, taxonomy and splits. Keep corpus, predictions and experimental checkpoints in ignored local paths; no photo/annotation/weight publication or cloud transfer is authorized. Any later commercial use, publication, transfer, expanded corpus or different purpose requires a fresh assessment. A known contrary restriction or rightsholder objection must be addressed rather than hidden by this record.
