# Implementation and existing environment

The scientific README and protocol are locked. Their hashes are checked by
`effect_pretraining.integrity.verify_release`; implementation instructions are kept here.

`resolved_environment.json` records inspected dependency paths and exact source/model
revisions. `scripts/run.py` creates ignored local pointers to those installed sources,
selects existing OSMesa libraries for CPU rendering, and runs commands through `uv`.
It does not install packages, change another checkout, or alter shared software.
The public checkpoint is loaded strictly using official LeRobot source. The nearby
AxisGuide model fork is excluded.

Run from the repository root:

```bash
python3 scripts/run.py -m pytest -q
python3 scripts/run.py -m ruff check src tests scripts
python3 scripts/run.py -m ruff format --check src tests scripts
```

The native correctness check additionally takes `--checkpoint` and `--vlm-assets`,
using the corresponding inspected paths in `resolved_environment.json`:

```bash
python3 scripts/run.py -m effect_pretraining.checks --checkpoint CHECKPOINT --vlm-assets VLM_ASSETS
```

The only current production execution entry point is the fixed numerical collector:

```bash
python3 scripts/run.py -m effect_pretraining.collection --artifacts ARTIFACT_DIRECTORY
```

It refuses to overwrite an existing collection marker. It records 100 native reset
attempts, ten fixed warm-up actions per attempt, capped contact setup on odd episodes,
and at most 200 retained actions per episode. It never retries to improve contact or
residual outcomes. Full artifacts belong in ignored `artifacts/` or another verified
storage location; small manifests and factual results belong in `audit/`.

`NumericalFlow` preserves native SmolVLA time/noise sampling, decoder/expert computation
and denoising, while adding a linear effect token after the state. Causal prefix blocks
and an attention padding mask isolate masked effects. Loss averages the seven physical
action coordinates. Only native action-generation weights are exported for transfer.
The shared effect transform and initial RMS calibration use complete nominal TRAIN
windows and a separate paired projector RNG. No training entry point may run before
both prescribed profiles, valid data, storage and resource admission are established.

The actual image path, production paired data/noise/mask streams and resource scheduling
still require integration validation before any neural campaign can be represented as
ready. Passing CPU checks does not replace those requirements.
