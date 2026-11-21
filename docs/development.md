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

The fixed numerical collector is:

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

`effect_pretraining.residuals` executes the fixed 400 reference replays and single ridge
diagnostic; `effect_pretraining.plots` renders its two-panel figure and source CSV.
`effect_pretraining.preparation` computes training-only effects and initial projector RMS.
These programs take the same `--artifacts` directory.

The acquisition runner, `effect_pretraining.training`, admits only complete measured
profiles and balanced six-fit 250-update rounds. It has not run a production update.
The S fit and final neural evaluation remain unvalidated integration work. The actual
image profile, production GPU behavior and resource projections must pass before any
neural campaign can be represented as ready. CPU checks do not replace those requirements.

The initial CUDA profile failed before its first update. Its failed record and platform
diagnostics are retained. After the prerequisite is repaired, an explicit
`--recover-pre-update-fault` flag on `effect_pretraining.profiling` preserves the failed
record before attempting the original profiles. It refuses to overwrite completed
profiles. It changes no schedule, condition, effective batch, objective or precision.

Raw official LIBERO observations are post-action. The S loader pairs observation row j
with actions[j+1:j+17] and excludes the final incomplete window. Both camera images use
the source-verified native 180-degree rotation and native SmolVLA resize/pixel processing.
Trainable action/state master parameters and Adam moments stay FP32 under BF16 autocast;
the frozen VLM keeps its native dtypes.
