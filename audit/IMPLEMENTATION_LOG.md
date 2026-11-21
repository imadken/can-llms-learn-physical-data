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
