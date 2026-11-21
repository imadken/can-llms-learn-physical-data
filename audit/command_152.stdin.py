import csv,json,os
from collections import Counter
from pathlib import Path
import numpy as np
from effect_pretraining.integrity import verify_release
root=Path.cwd();artifacts=root/'artifacts/stage1-v2-20261121';audit=root/'audit'
verify_release(root)
rows=list(csv.DictReader((audit/'baseline_episodes.csv').open()))
assert [(int(r['task_id']),int(r['initial_state_index'])) for r in rows]==[(t,e) for t in range(10) for e in range(20)]
assert all(not r['invalid_reason'] for r in rows)
steps=0
for row in rows:
 t,e=int(row['task_id']),int(row['initial_state_index']); n=int(row['control_actions'])
 assert 0<n<=300
 prefix=artifacts/'S_rollouts'/f'task{t:02d}_entry{e:02d}'
 assert prefix.with_name(prefix.name+'_initial.npz').is_file()
 with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as data:
  assert data['actions'].shape==(n,7)
  assert data['integration_states'].shape[0]==n
  assert all(np.isfinite(data[k]).all() for k in data.files)
 steps+=n
acq=list(csv.DictReader((audit/'acquisition_replays.csv').open()))
assert len(acq)==448 and all(r['status']=='valid' and int(r['control_actions'])==16 for r in acq)
assert len(list((artifacts/'acquisition_replays').glob('*.npz')))==448
res=list(csv.DictReader((audit/'residual_replays.csv').open()))
assert len(res)==400 and all(r['status']=='valid' for r in res)
logs=[json.loads(x) for x in (audit/'baseline_loss.jsonl').read_text().splitlines()]
assert [r['update'] for r in logs]==list(range(1,5001))
windows={tuple(w) for r in logs for w in r['window_ids']}
demos={(w[0],w[1]) for w in windows}
task_draws=Counter(w[0] for r in logs for w in r['window_ids'])
manifest=json.loads((audit/'demonstration_split_manifest.json').read_text())
permitted={(r['task_id'],x['trajectory_id']) for r in manifest for x in r['selected_trajectories'] if x['split']=='train'}
assert demos<=permitted and len(demos)==100
budget=json.loads((artifacts/'resource_budget.json').read_text())
jobs=[json.loads(x) for x in (artifacts/'accelerator_jobs.jsonl').read_text().splitlines()]
assert abs(sum(j['elapsed_job_seconds'] for j in jobs)-budget['accelerator_seconds_used'])<1e-6
assert not (artifacts/'active_accelerator_job.json').exists()
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0')
  if args[1:3]==[b'-m',b'effect_pretraining.campaign']:active.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active
result={'status':'passed','baseline_ordered_episodes':len(rows),'baseline_initial_snapshots':200,'baseline_control_actions':steps,'baseline_warmup_actions':2000,'baseline_invalid_episodes':0,'baseline_logged_updates':len(logs),'baseline_training_presentations':len(logs)*32,'baseline_unique_training_windows_observed':len(windows),'baseline_unique_training_trajectories_observed':len(demos),'baseline_task_training_draws':dict(task_draws),'acquisition_physical_chunks':len(acq),'acquisition_physical_actions':len(acq)*16,'residual_replay_records':len(res),'ledger_sum_matches_budget':True,'owned_campaign_processes_running':active,'active_reservation':False}
(audit/'episode_artifact_completeness.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
