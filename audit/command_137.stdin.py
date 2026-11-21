import json
from pathlib import Path
from datetime import datetime, timezone
from effect_pretraining.resources import read_budget
p=Path('artifacts/stage1-v2-20261121/resource_budget.json')
b=read_budget(p)
print(json.dumps({'as_of_utc':datetime.now(timezone.utc).isoformat(),'accelerator_hours_used':b.accelerator_seconds_used/3600,'accelerator_hours_remaining':(b.accelerator_limit_seconds-b.accelerator_seconds_used)/3600},indent=2))
