import json
from datetime import datetime,timezone
from pathlib import Path
from effect_pretraining.resources import read_budget
b=read_budget(Path('artifacts/stage1-v2-20261121/resource_budget.json'))
print(json.dumps({'as_of_utc':datetime.now(timezone.utc).isoformat(),'accelerator_hours_used':b.accelerator_seconds_used/3600,'remaining_hours':(b.accelerator_limit_seconds-b.accelerator_seconds_used)/3600}))
