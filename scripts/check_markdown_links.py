"""Check local destinations in tracked Markdown; private evidence stays private.

External URLs and heading fragments are reported out of scope, not fetched.
Missing private evidence is a warning; missing public repository destinations fail.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=('runs','work','exports','data','BANTAI-Datasets','.venv','.venv-training','.cache')


def destinations(text):
    text=re.sub(r'(?ms)^\s*(```|~~~).*?^\s*\1\s*$','',text)
    yield from (m[0] or m[1] for m in re.findall(r'\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+"[^"\n]*")?\s*\)',text))
    yield from re.findall(r'(?m)^\s*\[[^]\n]+\]:\s*<?([^>\s]+)>?',text)


def check(paths):
    result={'files':len(paths),'local_links':0,'public_missing':[],'private_missing':[],
            'private_links':0,'external_links':0,'fragments_checked':False}
    for path in paths:
        for target in destinations(path.read_text(encoding='utf-8')):
            if target.startswith('#'):continue
            if urlsplit(target).scheme:
                result['external_links']+=1;continue
            decoded=unquote(target.split('#',1)[0].split('?',1)[0])
            resolved=(ROOT/decoded.lstrip('/')) if decoded.startswith('/') else path.parent/decoded
            resolved=resolved.resolve();result['local_links']+=1
            try:relative=resolved.relative_to(ROOT)
            except ValueError:relative=None
            private=relative is None or (relative.parts and relative.parts[0] in PRIVATE)
            if private:result['private_links']+=1
            if not resolved.exists():
                result['private_missing' if private else 'public_missing'].append({'source':str(path.relative_to(ROOT)),'target':target})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extra',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    raw=subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),'ls-files','-z','--','*.md'],cwd=ROOT)
    paths={ROOT/p.decode('utf-8') for p in raw.split(b'\0') if p}
    paths.update(p.resolve() for p in args.extra)
    result=check(sorted(paths));payload=json.dumps(result,indent=2)
    if args.output:args.output.write_text(payload+'\n',encoding='utf-8')
    print(payload);return bool(result['public_missing'])

if __name__=='__main__':raise SystemExit(main())
