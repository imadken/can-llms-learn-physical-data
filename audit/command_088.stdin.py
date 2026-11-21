from pathlib import Path
import json
from effect_pretraining.resources import read_budget
for p in sorted(Path('audit').glob('acquisition_*_*.jsonl')):
 lines=p.read_text().splitlines();r=json.loads(lines[-1]);print(p.name,{'update':r['update'],'seconds':r['seconds']})
s=json.loads(Path('audit/acquisition_training.json').read_text());print('COMMON',s['common_completed_updates'],'COUNTS',{k:v['completed_updates'] for k,v in s['fits'].items()})
b=read_budget(Path('artifacts/stage1-v2-20261121/resource_budget.json'));print('ACCELERATOR_SECONDS_WITH_ACTIVE_JOB',round(b.accelerator_seconds_used,1))
