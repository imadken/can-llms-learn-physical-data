import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib
from libero.libero import benchmark, get_libero_path
from effect_pretraining.deployment import official_initial_states
from effect_pretraining.integrity import sha256_file
root=Path.cwd()
output=root/'audit/downstream_evaluation_bank.json'
if output.exists():
 raise FileExistsError('The fixed reset-bank manifest must not be overwritten')
suite=benchmark.get_benchmark_dict()['libero_goal']()
rows=[]
for task_id,task in enumerate(suite.tasks):
 states=official_initial_states(suite,task_id)
 state_path=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
 bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
 rows.append({'task_id':task_id,'task_name':task.name,'initial_states_path':str(state_path),'initial_states_sha256':sha256_file(state_path),'available_shape':list(states.shape),'dtype':str(states.dtype),'fixed_entry_indices':list(range(20)),'fixed_entries_sha256':hashlib.sha256(states[:20].tobytes()).hexdigest(),'bddl_path':str(bddl),'bddl_sha256':sha256_file(bddl)})
record={'created_utc':datetime.now(timezone.utc).isoformat(),'libero_revision':'8f1084e3132a39270c3a13ebe37270a43ece2a01','task_order':list(range(10)),'tasks':rows,'maximum_rollouts':200,'max_actions_per_rollout':300,'warmup_actions':10,'horizon':16,'actions_per_replan':4,'flow_integration_steps':10,'selection':'first 20 pinned official file entries; no outcome-based selection','demonstration_reset_disjointness':'not established; exploratory diagnostic'}
output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'pinned_before_S','tasks':len(rows),'maximum_rollouts':200,'shapes':[r['available_shape'] for r in rows]}))
