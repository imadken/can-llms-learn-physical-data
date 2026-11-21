# Execution plan and implementation decisions

The user selected https://github.com/imadken/can-llms-learn-physical-data.git at base `6af52ba`. Work is on `research/effect-pretraining-stage1-v2`. The supplied ZIP and existing AGENTS.md are preserved. All six protected-file SHA256 values were verified before extraction. The complete supplied release is preserved byte-for-byte; historical archive contents do not authorize execution.

## Plan fixed before experiments

1. Resolve installed CPU/GPU/software, native controller semantics, official model/checkpoint provenance and required data without modifying shared environments or other jobs.
2. Implement the locked numerical transforms, ideal command composition, temporary conditioning interface and native flow path. Validate scientific invariants and complete snapshot restoration before dependent collection/training.
3. Attempt the prescribed 100-episode corpus and Preliminary 1. Freeze episode/window/snapshot manifests. No outcome-dependent retries or selection.
4. Profile disposable numerical and actual multimodal paths (5 warm-up plus 20 measured updates each), inference and rollout cost. Preserve the fixed effective batches. Apply 25% projection contingency and reserve evaluation/checkpoint/package time.
5. Run acquisition only in balanced common 250-update rounds across P/K/D and seeds 101/202, up to 1,000 updates. Run the single fixed S schedule only when required data and measured resources permit. Respect eight accelerator-job hours, twelve hours after profiling starts, and the calendar cutoff.
6. Commit implementation and factual evidence, including incomplete/blocked components. Package the review bundle and external SHA256 checksum. No Stage 2, scientific changes, publication, job termination or automatic continuation.

## Discovery observations

- The selected repository initially contained only AGENTS.md; README.md, pyproject.toml and implementation/tests were absent.
- The command sandbox failed before execution (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`). Repository inspection required reviewed execution outside that broken sandbox.
- Hardware discovery found ARM Cortex-X925/Cortex-A725 CPU cores and NVIDIA GB10. Another process held approximately 66 GB of GPU memory. It was not modified.
- Existing environments and cached assets elsewhere are inspected as reusable installed dependencies only. No other project's checkout, scientific decisions, observations or datasets are substituted for this study.
- Locked files remain read-only. Verified runtime facts and calibration constants will be recorded separately.

## Verified implementation choices before collection

- Reused the existing Python 3.11.15 environment without installing or modifying shared packages. Official installed LeRobot 0.3.3 source and official LIBERO at 8f1084e3132a39270c3a13ebe37270a43ece2a01 are referenced through ignored local dependency pointers. The AxisGuide model fork is not used. The official LIBERO tracked worktree is clean.
- The public SmolVLA checkpoint revision and cached LFS weight hash were cross-checked via the public API. Strict loading matches all 450,046,176 parameters.
- Fixed headless CPU rendering discovery by selecting already-present user-space OSMesa libraries; no system libraries changed. Corrected two import/configuration entry-point mistakes before experiments.
- The independent SciPy float64 nominal rotation agrees with the controller's float32 quaternion implementation to 5e-8 absolute matrix tolerance; the initially tighter relative-only assertion was inappropriate near small entries. This is numerical validation tolerance, not an effect or attainment threshold.
- The geometry approach uses the compiled plate collision-box bounds and closed finger-pad midpoint, 1 mm fixed waypoint arrival tolerance, the prescribed +100 mm waypoint, +10 mm edge target, and at most 80 setup steps.
- Full integration snapshots include controller state, gripper persistent action, RNG, observation cache, robot buffers and control timing. Numeric JSON/NPZ serialization avoids executable pickle. A serialized snapshot produced a bit-identical 16-action replay in a separate engineering stream.
- Contacts will be recorded at every physics substep, not only at action endpoints. Native reset failures are not retried by the public wrapper.
- CPU native-model checks passed: nonzero effect-projector gradient through the frozen decoder, unchanged frozen parameter hashes after one disposable optimizer step, bit-identical actions after changing a masked effect, and finite ordinary language/state inference after removing the effect wrapper. This is not a production acquisition fit or multimodal profile.
- Current storage is insufficient for six ordinary uncompressed resumable checkpoints (7,189,150,464 bytes before overhead). Additional artifact storage was requested asynchronously; no shared files or jobs are deleted.

## Resumed after user-requested storage cleanup

The user requested a pause while cleaning disk space. Only this collector was suspended, at 85 completed attempts, and it resumed on the user's explicit instruction. No attempt was replaced. Disk free space increased to about 53 GB. The original collection completed all 100 attempts and 20,000 retained transitions with no invalid episodes. All 400 fixed replays completed; all 200 duplicate pairs were bit-identical in integration state. No robot-object contact was observed; this is retained as an observed collector limitation, with no enrichment retry.

The fixed ridge implementation explicitly minimizes mean squared error over records and the six target coordinates plus 0.001 times the squared coefficient norm, with an unpenalized intercept. Its loss-gradient test passed. All fitted effects/features use TRAIN episodes only.

The complete official raw LIBERO-Goal demonstration source was pinned at f13aa24a3da8c43c7225569f28c562979fa0e35a and all ten LFS file hashes were verified. Trajectory ordering was frozen before scores: SeedSequence([72026, task_id]) over sorted identifiers, taking 10 train and 5 validation per task.

The upstream LeRobot interface moved to src/lerobot/sim/integrations/libero.py. Its canonical observation at b13b952556c597d1e0849cadb6b6e7496bc5a2f3 verifies rotation by 180 degrees for both camera images and the native EEF position/axis-angle/two-finger state order. This source inspection does not upgrade the installed runtime.

## Raw demonstration timestamp repair before neural work

Official LIBERO create_dataset.py records observation[j] after executing action[j], while storing action[j] at the same row. A fixed five-row forward-kinematics audit on the first selected trajectory of task 0 confirmed that observations match the next recorded state much more closely (0.13–0.25 mm) than the same recorded state (3.0–5.6 mm). The loader therefore pairs observation row j with the complete next-action chunk actions[j+1:j+17], excluding the single incomplete terminal window. This corrects current-state/action alignment; it changes no trajectory allocation, controller command, condition, horizon or objective. The split manifest records this source-derived alignment explicitly, before profiling/training.

## Accelerator prerequisite fault and precision repair

The first required profile failed before update 0 while transferring the model to CUDA. No optimizer update or production fit occurred. Read-only diagnostics found approximately 37 GB reported free, one successful one-element CUDA allocation, and subsequent CUDA context initialization failures even before loading a model. Cgroup memory.max is unlimited. These observations establish a platform prerequisite failure; they do not prove a specific cause or implicate the scientific objective.

The matching GB10 context-initialization symptom is discussed in NVIDIA developer forum topic 384326. NVIDIA known-issues documentation describes a one-time cache flush as a debugging workaround. Host-wide cache flushing/compaction was proposed to the user for approval; no host control was changed without that approval, no other job was terminated, and no software upgrade or precision substitution occurred.

Source inspection also identified that AutoModel can preserve BF16 trainable expert weights from its config. Before any production update, the native expert and trainable projections are explicitly converted to FP32 master parameters, while using BF16 autocast and retaining native frozen dtypes. The CPU engineering check now verifies FP32 Adam moments. The earlier CPU disposable optimizer check is superseded for optimizer-precision evidence; no empirical run was affected. Calibration of the frozen state token is unchanged.

## Authorized platform cleanup and successful profiles

The user approved host cache cleanup and suggested pgx-status. That command reported healthy managed load but also the pre-existing root-owned VLLM engine using 66,019 MiB. It was not terminated. Sudo required a local password, so the user personally ran the proposed one-time sync/cache-flush/compaction command and confirmed completion. The original failed pre-update profile is preserved. The unchanged two 5+20-update profiles and the reserved development timing sample then completed under the original budget ledger. Profile weights are discarded; production starts from the pristine checkpoint and calibrated projector. No scientific schedule or treatment was changed.

## Permitted image microbatch optimization

The initial image profile used microbatch 1 with accumulation 32, averaged 2.224 seconds/update and peaked at 3.083 GB allocated. Its conservative full S schedule plus maximum-horizon evaluation projection would not fit after prioritized acquisition with the 25% contingency. Before any production result, choose microbatch 8 and accumulation 4, preserving the fixed effective batch 32, data/noise/time, objective and optimizer. This is explicitly permitted implementation optimization, not an additional scientific condition or S fit. Run one 5+20 disposable image timing check, preserve the original profile, reuse the already measured development inference/rollout timing, and count all added profiling time against the unchanged ceiling. No outcome metric selects this change.
