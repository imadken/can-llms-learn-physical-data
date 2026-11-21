import json
from pathlib import Path
from effect_pretraining.training import profiles_and_projection
base=Path('artifacts/stage1-v2-20261121')
profile=json.loads(Path('audit/profiling.json').read_text())
projection=profiles_and_projection(Path.cwd(),base)
print(json.dumps({'profile_status':profile['status'],'projected_seconds':projection,'actual_profile_job_budget':json.loads((base/'resource_budget.json').read_text()),'rollout_sample':profile.get('task_rollout_sample')},indent=2))
snaps=json.loads(Path('audit/snapshot_manifest.json').read_text())
selected=[]
for category in ['native_free_motion_attempt','contact_attempt']:
 selected.extend([s for s in snaps if s['status']=='valid' and s['intended_reset']==category][:16])
selected.sort(key=lambda s:(s['episode_id'],s['offset']))
for s in selected:
 s['reference_chunk_ids']=[0,1]
manifest={'selection':'first 16 valid snapshots per intended reset category in fixed Preliminary 1 order','snapshots':selected,'source_episode_ids':sorted(set(s['episode_id'] for s in selected)),'paired_training_seeds':[101,202],'generation_rng_seed':62026,'generation_rng_derivation':'SeedSequence([62026, training_seed, episode_id, snapshot_offset])','maximum_unique_generated_chunks_for_implemented_reuse':576,'maximum_physical_chunk_replays_for_implemented_reuse':448,'protocol_generated_chunk_ceiling':1024}
Path('audit/acquisition_bank_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
Path('audit/platform_remediation.json').write_text(json.dumps({'authorized_by':'user explicit approval in conversation','performed_by':'user in local terminal after sudo password requirement','commands':['sync','write 3 to /proc/sys/vm/drop_caches','write 1 to /proc/sys/vm/compact_memory'],'user_confirmed_completion':True,'pgx_status_observed':{'memory_health_status':'OK','managed_load_GiB':0,'preexisting_vllm_pid':996778,'preexisting_vllm_gpu_memory_MiB':66019},'other_jobs_terminated':False,'software_upgraded':False,'original_failed_profile_preserved':'audit/profiling_pre_update_fault.json','retry_result':profile['status']},indent=2)+'\n')
with Path('audit/IMPLEMENTATION_LOG.md').open('a') as f:
 f.write('\n## Authorized platform cleanup and successful profiles\n\nThe user approved host cache cleanup and suggested pgx-status. That command reported healthy managed load but also the pre-existing root-owned VLLM engine using 66,019 MiB. It was not terminated. Sudo required a local password, so the user personally ran the proposed one-time sync/cache-flush/compaction command and confirmed completion. The original failed pre-update profile is preserved. The unchanged two 5+20-update profiles and the reserved development timing sample then completed under the original budget ledger. Profile weights are discarded; production starts from the pristine checkpoint and calibrated projector. No scientific schedule or treatment was changed.\n')
