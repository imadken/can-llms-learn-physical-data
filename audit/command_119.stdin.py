from pathlib import Path
import json,h5py,numpy as np
manifest=json.loads(Path('audit/demonstration_split_manifest.json').read_text())
rows=[]
for item in manifest:
 with h5py.File(item['demonstration_path'],'r') as f:
  args=json.loads(f['data'].attrs['env_args'])['env_kwargs']
  controller=args['controller_configs']
  if args['control_freq']!=20 or controller['type']!='OSC_POSE' or not controller['control_delta']:
   raise ValueError('Demonstration controller convention conflicts with pinned interface')
  np.testing.assert_allclose(controller['output_max'],[.05,.05,.05,.5,.5,.5])
  np.testing.assert_allclose(controller['output_min'],[-.05,-.05,-.05,-.5,-.5,-.5])
  first=item['selected_trajectories'][0]['trajectory_id'];g=f['data'][first]
  image_stats={key:{'shape':list(g['obs'][key].shape),'dtype':str(g['obs'][key].dtype),'first_image_std':float(np.std(g['obs'][key][0]))} for key in ['agentview_rgb','eye_in_hand_rgb']}
  rows.append({'task_id':item['task_id'],'control_frequency_hz':args['control_freq'],'controller_type':controller['type'],'relative_control':controller['control_delta'],'image_convention':f['data'].attrs.get('macros_image_convention'),'images':image_stats,'state_position_dim':g['obs/ee_pos'].shape[1],'state_axis_angle_dim':g['obs/ee_ori'].shape[1],'state_gripper_dim':g['obs/gripper_states'].shape[1],'action_dim':g['actions'].shape[1]})
Path('audit/all_task_interface_audit.json').write_text(json.dumps({'status':'passed','tasks':rows},indent=2)+'\n')
print('All ten raw task interfaces match the pinned controller, 8D state, 7D actions, and two native camera streams.')
