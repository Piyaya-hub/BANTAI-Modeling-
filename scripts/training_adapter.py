"""Explicit training-only tensor adapter for the pinned HF 4.51.3 criterion.

No model subclassing, state registration, monkey patches, altered decoder inputs,
or inference integration. Optimizer ownership remains with the synthetic runner.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path

from .configuration import MODEL_ID, REVISION
from .training_smoke import LABELS, PROMPT, digest, validate_taxonomy

ADAPTER_VERSION = 'bantai.grounding-dino.encoder-output.v1'
EXPECTED_CONFIG = {'model_type':'grounding-dino','num_queries':900,'max_text_len':256,
    'two_stage':True,'auxiliary_loss':False,'embedding_init_target':True,
    'two_stage_bbox_embed_share':False,'decoder_bbox_embed_share':True,'decoder_layers':6,
    'class_cost':1.0,'bbox_cost':5.0,'giou_cost':2.0,'focal_alpha':.25,
    'bbox_loss_coefficient':5.0,'giou_loss_coefficient':2.0}


def adapter_source_digest(path=__file__):
    # Git can check out the same Python source with LF or CRLF on Windows.
    # Normalize only text line endings; retain sensitivity to every code change.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def config_signature(config):
    signature={key:getattr(config,key,None) for key in EXPECTED_CONFIG}
    if signature != EXPECTED_CONFIG:
        raise ValueError('Adapter supports only the assessed pinned architecture and native loss coefficients')
    return signature


def selected_proposals(outputs, queries):
    """Keep top-k ranking unchanged, retain gradients through gathered VALUES."""
    import torch
    logits=outputs.enc_outputs_class
    coords=outputs.enc_outputs_coord_logits
    if logits.ndim!=3 or coords.shape!=(*logits.shape[:2],4) or logits.shape[1]<queries:
        raise ValueError('Invalid encoder proposal shapes/capacity')
    indexes=torch.topk(logits.max(-1).values,queries,dim=1).indices
    selected_logits=logits.gather(1,indexes[:,:,None].expand(-1,-1,logits.shape[-1]))
    selected_boxes=coords.gather(1,indexes[:,:,None].expand(-1,-1,4)).sigmoid()
    if not torch.equal(selected_boxes.detach(),outputs.encoder_pred_boxes.detach()):
        raise ValueError('Selected proposal values differ from native decoder initialization')
    if outputs.init_reference_points.requires_grad or outputs.encoder_pred_boxes.requires_grad:
        raise ValueError('Native decoder reference-point detach boundary changed')
    if not torch.equal(selected_boxes.detach(),outputs.init_reference_points.detach()):
        raise ValueError('Decoder initial references differ from selected proposals')
    return selected_logits,selected_boxes,indexes


def validate_labels(labels, batch_size):
    import torch
    if not isinstance(labels,list) or len(labels)!=batch_size: raise ValueError('Target batch mismatch')
    for target in labels:
        ids,boxes=target['class_labels'],target['boxes']
        if ids.dtype!=torch.int64 or ids.ndim!=1 or boxes.shape!=(len(ids),4):
            raise ValueError('Target class/box shape or dtype mismatch')
        if not ((ids>=0)&(ids<len(LABELS))).all(): raise ValueError('Target outside approved taxonomy')
        if not torch.is_floating_point(boxes) or not torch.isfinite(boxes).all(): raise ValueError('Invalid target boxes')
        if len(ids):
            corners_min=boxes[:,:2]-boxes[:,2:]/2
            corners_max=boxes[:,:2]+boxes[:,2:]/2
            if not (boxes[:,2:]>0).all() or (corners_min < -2e-6).any() or (corners_max > 1+2e-6).any():
                raise ValueError('Target outside normalized padded image frame')


@dataclass(frozen=True)
class AdaptedOutput:
    outputs: object
    loss: object
    loss_dict: dict
    encoder_logits: object
    encoder_boxes: object
    selected_indexes: object


class EncoderOutputAdapter:
    """Stateless with respect to model weights: only immutable context is stored."""
    def __init__(self,model,tokenizer):
        if importlib.metadata.version('transformers')!='4.51.3': raise ValueError('Requires Transformers 4.51.3')
        validate_taxonomy()
        config_signature(model.config)
        self.token_ids=tuple(tokenizer(PROMPT)['input_ids'])
        self.tokenizer=tokenizer

    def manifest(self,model):
        import transformers
        library=Path(transformers.__file__).parent
        return {'adapter_version':ADAPTER_VERSION,'model_id':MODEL_ID,'revision':REVISION,
            'transformers':'4.51.3','configuration':config_signature(model.config),
            'prompt':PROMPT,'labels':list(LABELS),'token_ids':list(self.token_ids),
            'adapter_source_sha256':adapter_source_digest(),
            'adapter_source_hash_normalization':'UTF-8 with universal newlines',
            'runtime':{name:importlib.metadata.version(name) for name in ('torch','scipy','tokenizers')},
            'native_source_sha256':{p:digest(library/p) for p in
                ('models/grounding_dino/modeling_grounding_dino.py','loss/loss_grounding_dino.py','loss/loss_for_object_detection.py')},
            'decoder_reference_initialization':'native_detached_unchanged',
            'loss':'native_GroundingDinoForObjectDetectionLoss_unchanged_formula_coefficients',
            'resume_supported':False}

    def verify_manifest(self,model,manifest):
        if manifest != self.manifest(model):
            raise ValueError('Adapter context/source/configuration mismatch; requalification required')

    def __call__(self,model,batch):
        import torch
        from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps,build_text_mask
        from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
        config_signature(model.config)
        ids=batch['input_ids']
        if ids.ndim!=2 or ids.shape[0] not in (1,2): raise ValueError('Qualified batch size is 1 or 2')
        expected=torch.tensor(self.token_ids,dtype=ids.dtype,device=ids.device)
        if ids.shape[1]!=len(expected) or not torch.equal(ids,expected[None,:].expand_as(ids)):
            raise ValueError('Prompt/order/truncation differs from approved text mapping')
        if not torch.equal(batch['attention_mask'],torch.ones_like(batch['attention_mask'])):
            raise ValueError('Unexpected text padding for the fixed common prompt')
        validate_labels(batch['labels'],len(ids))
        # Do not request the model's miswired native loss. Its entire forward and
        # decoder remain untouched; call the existing criterion explicitly below.
        outputs=model(**{key:value for key,value in batch.items() if key!='labels'})
        logits,boxes,indexes=selected_proposals(outputs,model.config.num_queries)
        if model.training and torch.is_grad_enabled() and not boxes.requires_grad:
            raise ValueError('Undetached encoder regression path is missing')
        maps=build_label_maps(outputs.logits,ids)
        if any(row.shape[0]!=len(LABELS) for row in maps): raise ValueError('Class-span count changed')
        text_mask=build_text_mask(outputs.logits,batch['attention_mask'])
        loss,loss_dict,auxiliary=GroundingDinoForObjectDetectionLoss(outputs.logits,batch['labels'],outputs.logits.device,
            outputs.pred_boxes,model.config,maps,text_mask,encoder_logits=logits,encoder_pred_boxes=boxes)
        if auxiliary is not None: raise ValueError('Unexpected decoder auxiliary objective')
        if not torch.isfinite(loss).all() or not all(torch.isfinite(value).all() for value in loss_dict.values()):
            raise ValueError('Nonfinite corrected loss')
        return AdaptedOutput(outputs,loss,loss_dict,logits,boxes,indexes)
