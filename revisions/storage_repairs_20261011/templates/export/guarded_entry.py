import runpy,sys
from pathlib import Path
from diagnostic_guard import install
root=Path(sys.argv[sys.argv.index('--output-root')+1]);install(root)
entry=Path(__file__).with_name('candidate_quality.py');sys.argv[0]=str(entry)
runpy.run_path(str(entry),run_name='__main__')
