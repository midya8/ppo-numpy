"""Run the complete suite and audit commands against an installed wheel.

The copied src tree is inert input for existing hash/data tests, not an import
root. All ppolab modules must resolve inside the temporary environment before
and after tests/reports. Same-runtime replay is not independent training or
numerical evidence. Existing wheel/training/regeneration jobs stay in place.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['ppolab'] + ['ppolab.' + p.stem for p in sorted((ROOT / 'src/ppolab').glob('*.py')) if p.stem != '__init__']
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')


def call(args, cwd):
    return subprocess.check_output(args, cwd=cwd, env=ENV, text=True)


DRIVER = r'''
import importlib,importlib.util,json,pathlib,subprocess,sys
root=pathlib.Path.cwd().resolve()
source=pathlib.Path(sys.argv[1]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
names=json.loads(sys.argv[2])
for name in names:importlib.import_module(name)
def origins():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
    assert str(root/'src') not in sys.path
origins()
if sys.argv[3]=='tests':
    import pytest
    class Count:
        count=0
        def pytest_collection_finish(self,session):self.count=len(session.items)
    counter=Count()
    assert pytest.main(['-q',str(root/'tests')],plugins=[counter])==0
    assert counter.count>0
    print(json.dumps(dict(tests=counter.count)))
else:
    from ppolab.audit import gae_audit,ppo_audit
    from ppolab.clipface import audit
    from ppolab.clipverify import verify
    from importlib import resources
    data=resources.files('ppolab').joinpath('data')
    fixtures=json.loads(data.joinpath('fixtures.json').read_text())
    faces=json.loads(data.joinpath('clip-fixtures.json').read_text())
    assert len(fixtures)==26 and len(faces)==10
    reports=[]
    for f in fixtures:
        value=(gae_audit if f['kind']=='gae' else ppo_audit)(f['input'])
        assert value==f['report']
        reports.append((f['kind'],f['input'],value))
    for f in faces:
        x=f['input'];value=audit(x['old'],x['advantages'],x['epsilon'])
        assert value==f['report'] and verify(value)
        reports.append(('face',x,value))
    invocations=0
    def run(args):
        global invocations
        invocations+=1
        return subprocess.check_output(args,text=True)
    for i,(kind,payload,expected) in enumerate(reports):
        input=root/f'input-{i}.json';input.write_text(json.dumps(payload))
        report=root/f'report-{i}.json'
        if kind=='face':
            producers=[[sys.executable,'-I','-m','ppolab.clipface'],[str(prefix/'bin/ppo-clip-face')]]
        else:
            producers=[[sys.executable,'-I','-m','ppolab.audit',kind,'--input'],[str(prefix/'bin/ppo-audit'),kind,'--input']]
        for command in producers:
            text=run(command+[str(input)]);assert json.loads(text)==expected
            report.write_text(text)
            if kind=='face':
                for verifier in [[sys.executable,'-I','-m','ppolab.clipverify'],[str(prefix/'bin/ppo-clip-verify')]]:
                    assert run(verifier+[str(report)]).strip()=='verified declared clip-face certificate'
    assert invocations==112
    print(json.dumps(dict(api_reports=len(reports),command_invocations=invocations,entrypoints=6)))
origins()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        wheels = work / 'wheels'
        wheels.mkdir()
        call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '-w', str(wheels), str(ROOT)], work)
        built = list(wheels.glob('*.whl'))
        assert len(built) == 1
        # Reuse only dependency packages, never the checkout's ppolab package.
        venv.create(work / 'env', with_pip=True, system_site_packages=True)
        py = str(work / 'env/bin/python')
        call([py, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(built[0])], work)
        call([py, '-m', 'pip', 'install', 'pytest'], work)
        outside = work / 'outside'
        outside.mkdir()
        for name in ['tests', 'assets', 'scripts', 'docs', 'src']:
            shutil.copytree(ROOT / name, outside / name)
        for section in ['tests', 'commands']:
            if args.section in ['all', section]:
                print(call([py, '-I', '-c', DRIVER, str(ROOT), json.dumps(MODULES), section], outside))


if __name__ == '__main__':
    main()
