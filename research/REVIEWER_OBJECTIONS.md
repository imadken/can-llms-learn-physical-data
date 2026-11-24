# Anticipated reviewer objections

This is a scientific defense and interpretation guide for the supervisor. It does not authorize additional Codex experiments.

## 1. Is this only another pretraining ablation?

The question is whether retaining measured transition outcomes has value beyond training on the same actions and an ideal nominal motion summary. That comparison informs a data/compute allocation decision under an unchanged deployed policy. The contribution is the controlled empirical question and evidence, not the first inverse-dynamics or action-only objective. Explain this decision early, cite TAP/AR-VLA/PSR/SLIM directly, and avoid a sweeping audit of all published physics-aware methods.

## 2. Does D minus K simply equal the physical residual?

The input-coordinate difference is a residual under a declared nominal map. The downstream performance difference is not that residual, and does not isolate a unique causal physics contribution. D replaces K's supervision and changes its ambiguity, noise and difficulty. Explicitly retain these alternative explanations. The matched comparison estimates the effect of choosing that supervision procedure.

## 3. What if K is misspecified?

Publish and unit-check its command map. State that it composes ideal tracking from the initial pose. Do not feed it actual future controller targets or measured poses. Separate translation units, rotation conventions, timing, clipping and saturation. Report free-motion discrepancies as well as contact discrepancies. A faulty map requires correction; a declared ideal approximation requires bounded interpretation.

## 4. Why remove gripper displacement from the effect token?

The measured finger coordinate is not the same object as a gripper command or a persistent actuator target. The six-dimensional endpoint token makes nominal and measured conditioning comparable without an invented gripper dynamics model. Full current gripper state and gripper action targets remain in the model. Measured finger motion is logged. Accordingly, the claim concerns end-effector pose effects, not complete physical-state or gripper-dynamics supervision.

## 5. Why should the benefit survive removal of the numerical prefix?

The endpoint token is a compact motion-goal specification. Generating a chunk from that specification can teach the action expert to coordinate state-dependent actions over time using a conditioning signal. The native action expert and action-generation projections retain these computations when the temporary projector is discarded. Using the same action representation and flow objective during task fine-tuning makes reuse plausible. This is the mechanism hypothesis, not a proven equivalence between numeric and language goals. Acquisition plus the Stage 2 transfer contrasts tests the proposed path; the pilot does not prove it or uniquely identify the internal mechanism.

The token-scale rule prevents an arbitrary initialization amplitude from dominating this check: K and D share one nominal-training affine transform, and the paired initial effect projector matches the frozen state-token RMS. It does not guarantee semantic alignment or successful acquisition. No per-example normalization removes goal magnitude, and no evaluation outcome determines calibration.

## 6. Were contact or large residuals selected after seeing results?

The scene, contact-attempt reset mixture, excitation, episode counts and split IDs are fixed in advance. Failed contact attempts remain visible. Do not filter the corpus by residual magnitude or success. Report intended reset categories and realized contact fractions, and condition conclusions on this geometry-assisted collection regime. Privileged geometry is a collection resource, excluded from learned inputs.

## 7. Are K and D evaluated on the same objective?

K's native acquisition concerns its nominal map, while D's concerns measured endpoints. Present those metrics separately. The common physical-goal test supplies the same measured target to both and evaluates realized motion; for K it includes transfer of the nominal condition semantics. State this and report range coverage. Do not compare raw native training loss values as a common measure of skill.

## 8. Does changed output prove useful conditioning?

No. Pair the generation noise, use two reachable goals from the same initial snapshot, and score against both original and supplied goals. Action differences demonstrate sensitivity; lower error to the supplied goal is evidence of meaningful control. Null masking and alternative goals should not be confused with permanently training on false labels.

## 9. What does a weak or saturated S baseline tell us?

One seed and a short fixed schedule describe observed performance under that schedule. They do not identify a benchmark's ceiling or prove convergence. Zero success is not automatically evidence that the task has useful headroom. Report all tasks, counts and training duration; inspect implementation faults without searching for convenient scores. Later confirmatory schedules require a separate, documented resource/convergence decision.

## 10. Is a null result automatically valuable?

No. D approximately equal to K in a tiny pilot is descriptive. A meaningful Stage 2 negative conclusion requires valid supervision, evidence of acquisition, enough precision to exclude a declared benefit, and appropriate scope. If both K and D improve over P/S while D adds no meaningful benefit, nominal endpoint supervision may suffice under the tested conditions. If acquisition fails or intervals are wide, the question remains unresolved.

## 11. Why omit robustness and forgetting from the main claim?

They require different measurements and assumptions. Hidden mechanical parameters cannot be inferred at a decision from observations that do not distinguish them, although a prior or later observations might still support robustness. Representation probes also do not identify all information or a unique causal forgetting mechanism. These topics remain optional extensions; their omission keeps the central conclusion reachable.

## 12. What would each full-study outcome change in practice?

| Full-study outcome, with adequate evidence | Bounded practical implication |
| --- | --- |
| K improves over P and S; D adds no meaningful gain | Nominal endpoint supervision is sufficient for the observed benefit in this regime. |
| D improves over K/P/S and remains competitive with E | Measured endpoints justify additional supervision resources in this regime. |
| Adequately precise comparisons exclude the declared meaningful benefit over S for P/K/D; acquisition succeeded | These intermediate initializations do not justify their task-training cost under the tested transfer protocol. |
| D improves over S but E is better | Additional ordinary task updates are a better tested use of the matched compute. |
| Measured effects harm transfer | Report a failure of this supervision choice, accounting for ambiguity/difficulty and valid acquisition; do not infer that physics is generally harmful. |
| Acquisition fails, the map is invalid, or uncertainty is broad | No resolved answer; diagnose the specific limitation and avoid overclaiming. |

The scientific value lies in resolving a consequential distinction with sufficient evidence, not in guaranteeing that every possible pilot outcome is publishable.

## Final interpretation clarifications

A small residual in Preliminary 1 is not a downstream D/K null result. High ridge predictability is compatible with controller calibration but does not identify it causally; low ridge predictability may reflect missing observations or noise. A short K acquisition failure with passing correctness checks leaves acquisition unresolved, rather than proving that the pipeline is broken. The plate scene is prespecified and contact-enriched by attempted setup, not empirically established as the most favorable scene. Keep both primary transfer contrasts and qualify the scientific discussion to reflect the observed pilot evidence.
