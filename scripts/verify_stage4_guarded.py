"""Run Stage 4 software tests without model execution or private-artifact writes.

Generated fixtures/logs live in the system temporary directory. Training fixtures
are explicitly skipped; detector loading/forward, CUDA and updates are blocked.
"""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True


def walk(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from walk(item)
        else:yield item


def main():
    import torch
    from transformers import GroundingDinoForObjectDetection
    from scripts import train_taco as trainer
    def blocked(*args,**kwargs):raise AssertionError('Guard prohibits detector/CUDA/backward/optimizer execution')
    torch.set_num_threads(4)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    for test in walk(suite):
        try:source=inspect.getsource(getattr(test,test._testMethodName))
        except (TypeError,OSError):source=''
        if any(s in source for s in ('.backward(', 'autograd.grad(', 'gradient_paths(', 'train_step(', 'run_epoch(', 'train.main(', 'trainer.main(')) or test._testMethodName in (
            'test_epoch_loop_and_mid_epoch_resume_on_generated_images_with_toy_updates',
            'test_complete_optimizer_resume_matches_uninterrupted_update_and_rejects_tamper'):
            def skip():raise unittest.SkipTest('Guard excludes training-execution fixtures')
            setattr(test,test._testMethodName,skip)
    identity=trainer.source_identity()
    forbidden=[ROOT/name for name in ('runs','work','exports','data')]
    core={ROOT/name for name in trainer.SOURCES}|{ROOT/'config/taco_taxonomy.v1.json'}
    original_mkdir=Path.mkdir
    original_temp=tempfile.TemporaryDirectory
    def guarded_mkdir(path,*args,**kwargs):
        # Legacy fixture setup requests an already-existing work/ directory.
        # The actual TemporaryDirectory request below is redirected outside it.
        if path.resolve()==ROOT/'work' and kwargs.get('exist_ok'):return None
        return original_mkdir(path,*args,**kwargs)
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            flags=args[2] if len(args)>2 else 0
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if path in core or any(path.is_relative_to(p) for p in forbidden):raise AssertionError('Protected write: '+str(path))
            if path.is_relative_to(ROOT/'runs') and path.suffix.lower() in ('.pt','.bin','.safetensors'):
                raise AssertionError('Private checkpoint reads prohibited')
        elif event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink'):
            paths=args[:2] if event in ('os.rename','os.link','os.symlink') else args[:1]
            for value in paths:
                if isinstance(value,(str,bytes,os.PathLike)):
                    path=Path(os.fsdecode(value)).resolve()
                    if path in core or any(path.is_relative_to(p) for p in forbidden):raise AssertionError('Protected mutation: '+str(path))
    # A fresh verification process is required: Python audit hooks are permanent.
    sys.addaudithook(audit)
    log_folder=Path(tempfile.mkdtemp(prefix='bantai-stage4-guarded-'))
    def redirected_temp(*args,**kwargs):
        directory=kwargs.get('dir')
        if directory is not None and any(Path(directory).resolve().is_relative_to(p) for p in forbidden):
            kwargs['dir']=log_folder
        return original_temp(*args,**kwargs)
    with contextlib.ExitStack() as stack:
        for name in ('torch.optim.AdamW.step','torch.optim.SGD.step','torch.Tensor.backward',
                     'torch.autograd.backward','torch.autograd.grad','torch.cuda._lazy_init'):
            stack.enter_context(patch(name,blocked))
        stack.enter_context(patch.object(GroundingDinoForObjectDetection,'from_pretrained',blocked))
        stack.enter_context(patch.object(GroundingDinoForObjectDetection,'forward',blocked))
        stack.enter_context(patch.object(Path,'mkdir',guarded_mkdir))
        stack.enter_context(patch.object(tempfile,'TemporaryDirectory',redirected_temp))
        stack.enter_context(patch.object(trainer,'ROOT',log_folder))
        stack.enter_context(patch.object(trainer,'source_identity',return_value=identity))
        with (log_folder/'unit-tests.txt').open('w',encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    summary={'tests':result.testsRun,'passed':result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
             'failures':len(result.failures),'errors':len(result.errors),
             'skipped':[(test.id(),reason) for test,reason in result.skipped],
             'GPU_access':False,'detector_execution':False,'optimizer_updates':0,
             'protected_directory_writes':0,'logs':str(log_folder)}
    (log_folder/'unit-results.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
