import json
import subprocess
from pathlib import Path
from effect_pretraining.integrity import verify_release
root=Path.cwd()
verify_release(root)
audit=root/'audit'
entry=[]
for path in audit.glob('command_[0-9]*.json'):
 record=json.loads(path.read_text())
 if 'effect_pretraining.campaign' in record['argv']:
  entry.append({'command_record':path.name,'entry_code_commit':record['code_commit'],'started_utc':record['started_utc']})
core=['src/effect_pretraining/'+name+'.py' for name in ['baseline','training','acquisition','native_flow','training_data','deployment','simulation','coordinates','profiling']]
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for item in entry:
 item['current_head']=head
 item['science_core_unchanged_at_current_head']=not bool(subprocess.check_output(['git','diff',item['entry_code_commit'],head,'--',*core]))
record={'persistent_campaign_entries':entry,'resource_guard_note':'The process loaded the entry revision. The later nested reserve guard is not retroactive; the owned-process watchdog covers the active run.','historical_source_packaging':'Exact execution revisions included as source archives in the review bundle.'}
(audit/'execution_source_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
