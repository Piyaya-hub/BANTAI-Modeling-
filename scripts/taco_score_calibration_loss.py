"""D067 differentiable auxiliary objective; native adapter and criterion untouched."""
import time


def class_scores(logits,input_ids):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    if logits.ndim!=3 or logits.shape[:2]!=(1,900):raise ValueError('Require one image /900 queries')
    maps=build_label_maps(logits,input_ids)
    if len(maps)!=1 or maps[0].shape!=(8,logits.shape[-1]):raise ValueError('Require eight ordinary label spans')
    spans=maps[0].bool()
    if not spans.any(-1).all() or (spans.sum(0)>1).any():raise ValueError('Empty/overlapping class spans')
    probs=torch.stack([logits[:,:,span].sigmoid().mean(-1) for span in spans],-1)
    means=torch.stack([logits[:,:,span].mean(-1) for span in spans],-1)
    if not torch.isfinite(probs).all() or not torch.isfinite(means).all():raise ValueError('Nonfinite ordinary class span')
    return probs,means


def pairwise_iou(predicted,targets):
    """Valid-image normalized cxcywh on both sides (no padded-frame conversion)."""
    import torch
    def corners(boxes):return torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
    a,b=corners(predicted),corners(targets)
    wh=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp(min=0)
    inter=wh.prod(-1);union=predicted[:,2:].prod(-1)[:,None]+targets[:,2:].prod(-1)[None,:]-inter
    return torch.where(union>0,inter/union.clamp(min=1e-12),torch.zeros_like(inter))


def calibration_losses(logits,boxes,input_ids,target):
    import torch
    from .training_adapter import validate_labels
    validate_labels([target],1)
    if boxes.shape!=(1,900,4) or not torch.isfinite(boxes).all():raise ValueError('Invalid calibration boxes')
    probs,means=class_scores(logits,input_ids)
    if not len(target['class_labels']):raise ValueError('Empty training GT is not authorized')
    with torch.no_grad():
        overlaps=pairwise_iou(boxes[0].detach(),target['boxes'])
        scores=overlaps+probs[0,:,target['class_labels']].detach()
        selected=scores.argmax(0)
    zero=probs.sum()*0.;pos=zero;absent=zero
    weights=(1.5,.75,.75,.75,0.,.75,.75,.75);assignments=[]
    present=set(target['class_labels'].tolist())
    for j,(cls,q) in enumerate(zip(target['class_labels'].tolist(),selected.tolist())):
        p=probs[0,q,cls];pos=pos-weights[cls]*torch.log(p+1e-6)
        assignments.append({'GT_index':j,'category_id':cls,'query_id':q,'p':float(p.detach()),
                            'IoU':float(overlaps[q,j]),'weight':weights[cls]})
    active={}
    for cls in range(8):
        if cls in present:continue
        values=probs[0,:,cls];values=values[values>.03]
        active[str(cls)]=len(values)
        if len(values):absent=absent-10.*(torch.log(1.-values+1e-6)*values.square()).mean()
    if not torch.isfinite(pos+absent):raise ValueError('Nonfinite auxiliary calibration loss')
    return pos,absent,{'positive_assignments':assignments,'absent_class_passing_queries':active,
                       'class_probs_shape':list(probs.shape),'class_logits_shape':list(means.shape)}


def calibrated_train_step(model,adapter,optimizer,batch):
    """One adapter forward/backward/update with original gradient and clip audits."""
    import torch
    from .training_smoke import checked_loss
    model.train();optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
    result=adapter(model,batch)
    components=checked_loss(result,model.config)
    pos,absent,audit=calibration_losses(result.outputs.logits,result.outputs.pred_boxes,batch['input_ids'],batch['labels'][0])
    total=result.loss+pos+absent
    if not torch.isfinite(total):raise ValueError('Nonfinite augmented loss')
    total.backward()
    probes=('model.enc_output.weight','model.encoder_output_bbox_embed.layers.2.weight')
    named=dict(model.named_parameters());gradients={}
    for name,p in named.items():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):raise ValueError('Missing/nonfinite gradient: '+name)
    before={name:named[name].detach().clone() for name in probes}
    for name in probes:
        gradients[name]=float(named[name].grad.norm())
        if gradients[name]<=0:raise ValueError('Expected nonempty encoder gradient: '+name)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    updates={name:float((named[name].detach()-before[name]).abs().max()) for name in probes}
    if not all(v>0 for v in updates.values()):raise ValueError('Encoder update missing')
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):raise ValueError('Nonfinite updated parameter')
    return {'loss':float(total.detach()),'native_loss':float(result.loss.detach()),'loss_pos':float(pos.detach()),
            'loss_absent':float(absent.detach()),'calibration':audit,'components':components,
            'gradient_norm_before_clip':float(norm),'encoder_gradient_norms':gradients,'encoder_update_max_abs':updates,
            'seconds_including_gradient_audit':time.perf_counter()-start}
