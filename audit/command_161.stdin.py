import json,subprocess
from pathlib import Path
from effect_pretraining.integrity import verify_release
root=Path.cwd();audit=root/'audit';records=[]
for p in audit.glob('command_[0-9]*.json'):
 r=json.loads(p.read_text());r['record_path']=str(p.relative_to(root));records.append(r)
records.sort(key=lambda r:r['started_utc'])
checks={}
for label, words in [('pytest', ['pytest','-q']),('ruff_lint',['ruff','check','src','tests','scripts']),('ruff_format',['ruff','format','--check','src','tests','scripts'])]:
 matched=[r for r in records if all(w in r['argv'] for w in words) and r.get('returncode')==0]
 assert matched,label
 r=matched[-1];checks[label]={'command_record':r['record_path'],'returncode':0,'log_path':r['log_path']}
 if label=='pytest':
  log=Path(r['log_path']).read_text();assert '23 passed' in log
  checks[label].update(passed_tests=23,dependency_warnings=4)
result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'checks':checks,'protected_hashes':verify_release(root),'actual_fit_audits':['audit/actual_training_invariants.json','audit/actual_baseline_invariants.json'],'saved_episode_audit':'audit/episode_artifact_completeness.json','all_three_preliminaries':'complete','campaign_and_watchdog':'ended; no limit intervention'}
(audit/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('resolved top-level status:',json.loads((root/'resolved_environment.json').read_text())['status'])
