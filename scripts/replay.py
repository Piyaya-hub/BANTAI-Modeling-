"""Source-verified saved-result replay in one shared GUI tab, without inference."""

import copy
import hashlib
from pathlib import Path
import sys

from .configuration import resolve_path,strict_load
from .inputs import load_image
from .persistence import atomic_json,create_run
from .results import validate_result
from .viewer import DisplayBatch,safe_text,show_image,summary


def verified_source(result_path,image_path):
    result_path=resolve_path(str(result_path));image_path=resolve_path(str(image_path))
    before=result_path.read_bytes()
    result=validate_result(strict_load(result_path))
    if result_path.read_bytes()!=before:raise ValueError("Result JSON changed during validation")
    image,source,error=load_image(image_path)
    if image is None:raise ValueError(f"Replay original cannot be decoded: {error['code']}")
    try:
        for key in ("image_id","sha256","width","height","coordinate_frame"):
            if source[key]!=result["source"][key]:
                raise ValueError(f"Replay source mismatch: {key}; use the matching unchanged original")
    except BaseException:
        image.close()
        raise
    return image,result,result_path,before


def run_replay(config,result_path,image_path):
    result_paths=list(result_path) if isinstance(result_path,(list,tuple)) else [result_path]
    image_paths=list(image_path) if isinstance(image_path,(list,tuple)) else [image_path]
    if not result_paths or len(result_paths)!=len(image_paths):
        raise ValueError("Replay requires equal nonzero numbers of --result and --image, paired in order")
    sources=[];attempts=[];batch=DisplayBatch()
    try:
        # Validate every pair before any pending attempt or window is created.
        for rp,ip in zip(result_paths,image_paths):sources.append(verified_source(rp,ip))
        for number,(image,result,rp,before) in enumerate(sources,1):
            shown=copy.deepcopy(result)
            enabled=config["display"]["enabled"]
            policy="passing_only" if enabled or config["display"]["terminal_mode"]=="summary" else "full_raw"
            shown["display"].update(config["display"],console_policy=policy)
            run_id,folder=create_run(config["paths"]["results"]);path=folder/"replay.json"
            record={"schema_version":"bantai.display-replay.v1","run_id":run_id,
                "inference_executed":False,"uses_saved_predictions":True,"image_number":number,
                "source_result":{"path":str(rp),"sha256":hashlib.sha256(before).hexdigest(),"run_id":result["run_id"]},
                "source":result["source"],"model_from_saved_result":result["model"],
                "thresholds_from_saved_result":{k:result["inference"][k] for k in ("threshold","text_threshold")},
                "display":dict(config["display"],status="pending" if enabled else "disabled",error=None,
                               visual_verification="pending",console_policy=policy)}
            atomic_json(path,record)
            attempts.append((path,record,image,shown))
        try:
            if config["display"]["enabled"]:
                try:
                    for path,record,image,shown in attempts:show_image(image,shown,batch=batch)
                    batch.show()
                except Exception as exc:
                    for path,record,_,_ in attempts:
                        record["display"].update(status="error",error={"stage":"display","code":"viewer_failed","message":str(exc)})
                        atomic_json(path,record,replace=True)
                    print(f"Replay display failed; saved predictions preserved: {safe_text(str(exc))}",file=sys.stderr)
                else:
                    for path,record,_,_ in attempts:
                        record["display"]["status"]="shown"
                        atomic_json(path,record,replace=True)
            else:
                for _,_,_,shown in attempts:
                    print(f"IMAGE: {safe_text(Path(shown['source']['image_id']).name)}")
                    print("Image display deliberately disabled; source identity verified.")
                    print(summary(shown))
        except KeyboardInterrupt:
            print("Replay interrupted; pending display attempts remain unfinished.",file=sys.stderr)
            return 130
        for path,_,_,_ in attempts:print(f"Display-only replay JSON: {path}")
        return 1 if any(record["display"]["status"]=="error" for _,record,_,_ in attempts) else 0
    finally:
        batch.close()
        for image,_,_,_ in sources:image.close()
