import os
from pathlib import Path
from diagnostic_guard import fresh_contract_rename
import subprocess,sys

def test_only_fresh_journal_contract_allowed(tmp_path):
    root=tmp_path.resolve();source=root/('contract.'+'a'*32+'.pending');source.write_text('{}')
    destination=root/'contract.json'
    assert fresh_contract_rename(root,(str(source),str(destination),-1,-1))
    destination.write_text('old')
    assert not fresh_contract_rename(root,(str(source),str(destination),-1,-1))
    assert destination.read_text()=='old'

def test_other_moves_and_directory_handles_rejected(tmp_path):
    root=tmp_path.resolve();source=root/('contract.'+'b'*32+'.pending');source.write_text('{}')
    assert not fresh_contract_rename(root,(str(source),str(root/'other.json'),-1,-1))
    assert not fresh_contract_rename(root,(str(source),str(root/'contract.json'),4,-1))
    assert not fresh_contract_rename(root/'subdir',(str(source),str(root/'contract.json'),-1,-1))

def test_actual_journal_commit_and_reopen_under_guard(tmp_path):
    remote=Path('@PROJECT_ROOT@/releases/qwen_minimal_peer_reuse_r6_20261001/source')
    local=Path(__file__).resolve().parent.parent/'live_qwen_llava/qwen_minimal_peer_reuse_r6_20261001/source'
    source=remote if remote.exists() else local
    code="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from checkpoint_journal import Journal
from diagnostic_guard import install
root=Path(sys.argv[2]);install(root)
with Journal(root/'journal',{'test':'fresh'}) as j:j.commit('sample',{'tokens':[1,2]})
with Journal(root/'journal',{'test':'fresh'}) as j:assert j.result('sample')=={'tokens':[1,2]}
print('passed')
"""
    result=subprocess.run([sys.executable,'-B','-c',code,str(source),str(tmp_path)],
        cwd=Path(__file__).resolve().parent,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'passed' in result.stdout
