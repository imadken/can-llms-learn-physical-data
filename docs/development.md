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
profiles and balanced six-fit 250-update rounds. The persistent entry point
`effect_pretraining.campaign` keeps one CUDA context alive through those rounds,
the fixed acquisition evaluation and the admitted S fit/evaluation. The full reservation,
including CPU replay between accelerator phases, counts once against the ceiling.
Actual completion is recorded in `audit/status.json` and the per-component JSON/CSV
outputs; successful CPU checks alone do not establish production completion.

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

The final read-only checkpoint and paired-stream audits are:

```bash
python3 scripts/run.py -m effect_pretraining.training_audit --require-final
python3 scripts/run.py -m effect_pretraining.training_audit --baseline --require-final
python3 scripts/run.py -m effect_pretraining.plots --artifacts ARTIFACT_DIRECTORY --component neural
python3 scripts/run.py -m effect_pretraining.reporting --artifacts ARTIFACT_DIRECTORY
```

The baseline audit is used only if S completed; partial or deferred components must
retain their actual status. Reporting can summarize an interrupted evaluation's
existing episode CSV without additional rollouts, and distinguishes unperformed
episodes from measured failures.

`scripts/watch_campaign.py` is an immediate, one-shot backup for the active campaign's
hard deadline. It verifies the owned process and reservation identity, exits when the
campaign exits, and creates no scheduled continuation. It does not control unrelated
processes. The internal reserve guard also refuses a new phase within the final
180 seconds; the external guard covers the already-loaded campaign revision.

After committing implementation and small results, run `scripts/package_review.py`
directly through the existing `uv run --no-project --no-config` interpreter with
`PYTHONPATH=src`. Do not wrap packaging with `scripts/run.py`, because that logger
would create an uncommitted command record during the clean-result check. The bundle
records committed code/results revisions and full-artifact hashes, includes the five
fixed replay examples, and receives an external SHA256 file. Unrelated worktree files
are preserved. No remote push is part of this workflow.
