# Stage 1 scope and claims

Version 2.2-locked, 20 November 2026. This is the frozen author plan for RSS Stage 1. No empirical result is claimed. See RESEARCH_LOCK.json at the package root for the protected files and hashes.

## 1. The scientific argument stays central

RSS 2027 permits missing or preliminary empirical validation at Stage 1 while requiring a developed problem, hypotheses and methodology. Its review criteria emphasize the value of the hypothesis and feasibility of a defensible conclusion [R1-R3]. Our three preliminaries demonstrate exploration and technical progress. They do not decide the paper's scientific question, and they do not have to be positive to be reported.

The submission remains five content pages and one reference page. The initial deadline is 4 December 2026 at 23:59 AoE, equivalent to 5 December at 12:59 in Paris. Repository links and audit bundles are for collaboration, not substitutes for evidence inside the anonymous submission [R1].

## 2. Limited revisions to the previous plan

| Element | Current decision | Reason |
| --- | --- | --- |
| Headline | What transfers from effect-conditioned action pretraining? | Put both nominal conditioning and measured outcomes inside one question. |
| Central contrasts | K versus P and D versus K | Both are already available from the planned conditions; no new training arm is introduced. |
| Effect token | Six-dimensional end-effector pose effect | Define the nominal control without inventing a model of hidden gripper actuator state. Full current proprioception and gripper action remain. |
| Physical shifts | Optional Stage 2 extension | Robustness to hidden mechanics is a separate question. No parameter sweep in the preliminaries. |
| Retention/forgetting probes | Optional Stage 2 extension | They are not needed to state the primary transfer comparison. |
| Two downstream demonstration budgets | Retained for Stage 2 | A middle budget was already optional. |
| Preliminary experiments | Residual characterization, small acquisition, one S baseline | Three bounded pieces of exploratory evidence with no downstream D/K/P performance campaign. |

The frozen VLM, temporary effect interface, native flow objective, transfer through action-generation weights, S/P/K/D/E conditions and identical deployment interface remain.

## 3. Title, question and intended contributions

**Title:** What Transfers from Effect-Conditioned Action Pretraining to Vision-Language-Action Policies?

**Question:** With interaction trajectories, architecture and optimization controlled, what downstream value comes from learning the action distribution, conditioning on a nominal end-effector effect, and conditioning on its measured counterpart?

**Decision informed:** Given numerical interaction data and a limited training budget, a practitioner can train an action prior, condition on nominal endpoints, condition on measured endpoints, or allocate the same compute to the available task demonstrations. The P/K/D/S/E comparisons estimate which choice provides useful task transfer in the declared regime. A well-resolved favorable or unfavorable result informs that choice; an inconclusive result is reported as unresolved.

**Contributions:**

1. A controlled comparison that distinguishes the transfer value of an action prior, nominal effect conditioning and measured effect conditioning under one deployed policy architecture.
2. A concrete intermediate-training procedure that uses the native action-generation objective and discards the temporary effect input before task learning.
3. An evaluation and interpretation protocol that connects acquisition to downstream success, practical compute allocation and bounded negative conclusions.

Let J(m) denote the expected mean task success of method m under the declared demonstration budget, task suite and evaluation protocol. This is an empirical comparison of supervision choices. P, K and D are not a nested information hierarchy. D replaces K's token; it does not append a residual to a common nominal token. Token noise, ambiguity, conditional entropy and optimization difficulty can change. The arithmetic identity J(D)-J(P) = [J(K)-J(P)] + [J(D)-J(K)] is not a unique causal decomposition of physics and intent.

The phrase 'pure contribution of physics' is excluded. Use 'additional value of measured end-effector effects under the declared training procedure.'

## 4. Motivation and mechanism hypothesis

The endpoint effect serves as a compact motion-goal specification. To generate an action chunk that realizes that goal, the action expert can learn to use conditioning information to coordinate state-dependent actions across time. The relevant conditional computations are learned in the native action expert and its action-generation projections, which are retained after the temporary effect projector is removed. Subsequent task fine-tuning uses the same action representation and flow objective, allowing the retained generator to adapt from numerical motion goals to goals conveyed by the image-language prefix. We hypothesize that this initialization improves target-task learning beyond the action-distribution prior; measured endpoints may supply additional supervision beyond nominal command composition.

This hypothesis concerns reusable conditional action generation. It does not assume that numerical effects and image-language prefixes have identical semantics or that the frozen VLM automatically aligns them. The later prefix distribution changes, so useful transfer may fail. Stage 2 first evaluates acquisition, then tests whether the retained weights improve downstream learning. K versus P tests nominal goal conditioning, and D versus K tests the additional value of measured supervision. These comparisons do not uniquely identify every internal mechanism.

## 5. Direct related work and scope of novelty

TAP already studies task-agnostic inverse-dynamics pretraining followed by task alignment [R4]. AR-VLA already studies independent action-expert pretraining without image-language supervision [R5]. PSR and SLIM are close predictive sensorimotor/action representation precedents [R6-R7]. SmolVLA supplies the existing flow-matching action interface [R8-R9].

We do not claim the first task-free, image-free, inverse-dynamics or physics-informed pretraining method. Our contribution is the controlled supervision comparison and its implications for unchanged deployment. PSR's abstract and bibliographic identity were reconfirmed on 20 November; direct retrieval of its full PDF/HTML and project page remained unavailable. The comparison stays limited to the primary abstract. No stronger priority claim or statement about unverified PSR ablations is permitted without reading its full text.

## 6. Information and method

Let x_t denote the full simulator state, r_t the available robot proprioception, A_t the H-action chunk, p_t end-effector position and R_t its rotation. The model's r_t is not assumed to be a complete Markov state. In the chosen LIBERO interface, retain the native eight-dimensional proprioception and seven-dimensional actions, including gripper coordinates and commands [R10].

The current measured effect is:

$$e_t^D=[p_{t+H}-p_t,\ \operatorname{Log}(R_t^\top R_{t+H})]\in\mathbb{R}^{6}.$$

The nominal effect is e_t^K = G(r_t,A_t), computed by ideal tracking of the pinned controller's declared command interpretation. G uses the initial observed pose and commanded actions only. It includes input/output scaling, clipping, world/local rotation convention and declared limits, but uses no future measured pose, hidden mechanics, object state or actual controller target trajectory.

When actual commands are defined relative to achieved poses, ideal command composition is not identical to the controller targets produced during the real replay. This distinction must be named and validated. K is a transparent nominal model, not a ground-truth physical baseline. The gripper action is still predicted and current gripper state remains available, but no nominal finger-displacement target is invented. Actual finger motion and contacts are diagnostic metadata, not effect-token coordinates.

Use native flow matching with X_tau=(1-tau)A_t+tau*epsilon and velocity target epsilon-A_t. Retain the native noise/time distribution. During intermediate training, freeze the VLM and state projector; train the action expert, native action/time/output projections and one temporary linear 6-to-prefix-width effect projector. Use the same neutral token allocation in P and paired 20% effect attention masking across P/K/D. Gradients must reach the effect projector through the frozen decoder.

Before neural training, define one common affine effect transform from nominal effects in the fixed training split, with fixed physical scale floors. Apply it identically to K and D. Match the effect projector's initial embedding RMS to the frozen native state-token RMS once, using training inputs only and the same paired initialization for P/K/D. Absorb the scale into the initial projector weights; do not normalize each sample's effect magnitude or calibrate from evaluation scores. The precise deterministic rule is in the implementation protocol.

Remove the effect projector before ordinary task fine-tuning. Transfer only native action-generation weights, reset optimizers, and use the ordinary images/instruction/current-state interface. Train the native state projector identically in all downstream arms while keeping the VLM frozen. Deployed architecture, inputs, horizon and control settings are identical.

## 7. Conditions and hypotheses

| Arm | Intermediate treatment | Downstream role |
| --- | --- | --- |
| P | Same numerical state/action data, neutral effect slot | Action-distribution prior |
| K | Same data, ideal nominal endpoint token | Nominal conditioning |
| D | Same data, measured endpoint token | Measured conditioning |
| S | No additional intermediate training | Standard practical anchor |
| E | No intermediate training; extra ordinary task updates | Compute-allocation anchor |

Two prespecified low-data contrasts form the primary family:

- C1 = J(K)-J(P): the value of nominal endpoint conditioning beyond an action prior.
- C2 = J(D)-J(K): the additional value of measured endpoint conditioning.

The directional hypotheses propose positive gains, with five percentage points as the provisional smallest effect of practical interest for each. The study estimates either direction; it is not accepted or rejected by whether D wins. This practical margin is a scientific planning choice, not an RSS rule. Freeze the confirmatory margin before results from that study.

Comparisons with S and E establish practical usefulness. The low-versus-full demonstration interaction remains secondary. No pilot D/K/P task fine-tuning is included in the initial execution brief.

## 8. Defined Stage 2 core

Retain one SmolVLA checkpoint, the full ten-task LIBERO-Goal suite, one multitask policy per fit and two demonstration budgets. The proposed low/full allocations remain 10/40 training demonstrations per task after a dataset audit, with separate validation. All intermediate arms share a frozen numerical corpus and normalization conventions. Acquisition is evaluated before downstream transfer is interpreted.

The previous planning allocation of eight paired pipeline seeds gives 24 intermediate fits and 80 downstream fits, with 40,000 nominal task rollouts at 50 reset states per task/policy. These are planning quantities, not authorization to launch and not a guarantee of sufficient precision. Profile the DGX Spark and determine a fixed confirmatory replication count from independent variance and resource evidence. No 32,000-rollout physical-shift campaign is in the mandatory core now.

Within each pipeline seed, equally average the ten tasks and form both paired contrasts. Inference concerns the fixed suite/corpus/evaluation bank and declared training randomness. Do not use evaluation episodes as independent training replicates. Construct a 97.5% marginal two-sided confidence interval for each of the two primary contrasts. By Bonferroni, the pair has at least 95% simultaneous coverage under the stated interval assumptions. Use separate prespecified multiplicity control for any formal secondary family. Report ordinary point estimates and uncertainty clearly.

A precise interval excluding the declared useful benefit is informative only when acquisition and comparison validity are established. A wide interval is inconclusive. Near equality at preliminary scale is not equivalence. A favorable probe is not a substitute for task utility. Greater performance relative to P alone is insufficient if the method underperforms S.

Mechanical shifts, retention probes, replay interventions, another architecture or hardware are possible Charter extensions. They are not an automatic response to an interesting or unfavorable pilot result.

## 9. Proposed Extension Charter and five-page story

Commit to: lock the implementation/data; validate nominal versus measured effects; evaluate acquisition; execute the five-arm, two-budget nominal study with fixed replication; report task and cost outcomes, uncertainty and all deviations; release enough evidence for numerical review; draw bounded conclusions. The eventual Charter remains subject to reviewer and chair agreement.

Keep the original five-page allocation: introduction 0.70; direct related work 0.45; formulation/hypotheses 0.75; method/controls 1.35; evaluation/interpretation 1.50; scope 0.25. Preliminary evidence receives at most approximately half a page within the existing allocation. If three panels make the page crowded, include one compact residual/acquisition figure and an S baseline sentence or small table. Do not sacrifice the scientific argument to display every pilot plot.

The paper's concise Stage 2 statement is: first establish acquisition of the declared endpoint-conditioning task, then evaluate the five training conditions at the prespecified demonstration budgets with an identical deployed policy interface. Paired training replicates and a prespecified practical margin distinguish useful transfer, limited additional benefit and unresolved evidence. Acquisition checks and the S/E anchors determine which interpretation is justified. The full replicate allocation, interval construction, numerical margins and execution details remain in the research protocol; retain enough summary in the paper to make the conclusion's feasibility assessable.

One compact, honestly labelled preliminary figure is sufficient. Completed weak or negative evidence remains observed evidence and must not be relabelled as an unperformed experiment. Unfinished work is described as planned. If no valid evidence is audited by the cutoff, submit the scientific argument and defined study without invented empirical results.

## 10. Revision rule and outcome-dependent interpretation

Small, negative, mixed or exceptionally favorable preliminary findings do not authorize a change of question, task selection, arm allocation, outcome metric, or paper scope. Record them, interpret within their limits, and proceed only with already specified work.

This rule fixes the research question and comparisons; it does not freeze the empirical interpretation. Preliminary evidence can change how the motivation's assumptions are qualified and how limitations are discussed. K versus P and D versus K remain the two prespecified contrasts regardless of the residual observation. A small endpoint residual is not a bounded null in downstream transfer, and a large residual does not establish useful transfer.

| Preliminary observation | Permitted interpretation |
| --- | --- |
| Residual appreciable relative to replay variability and declared physical tolerances | The two endpoint-supervision choices differ in the sampled development regime. |
| Small residual | Limited endpoint separation in this collector; downstream equivalence is untested. |
| Ridge predicts much of the residual | Simple predictable structure is present; controller calibration and other correlated effects remain possible explanations. |
| Ridge predicts little | This linear diagnostic is insufficient; nonlinearity, missing observations and noise remain possible. |
| Native K acquisition meets the descriptive criterion | The nominal-conditioning path learned usable conditioning within the pilot. |
| Native K acquisition is weak while correctness checks pass | Acquisition remains unresolved within the fixed budget; this alone does not prove a broken pipeline or falsify transfer. |
| D acquisition meets the descriptive criterion | Measured endpoint conditioning is operational in the pilot; downstream superiority remains untested. |
| A mapping, gradient, masking or replay check fails | A specific implementation or measurement defect needs correction and affected results must be identified. |

The plate scene is described as one prespecified development scene with contact-attempt enrichment. Neither 'best-case' nor representativeness across LIBERO-Goal is established by this pilot. A large free-motion residual must be discussed as including nominal-map/controller tracking differences. The existing ridge diagnostic does not identify those causes. No calibrated extra training arm is introduced.

The user has now frozen the author plan. Neither the supervisor nor Codex may autonomously change its scientific question, comparisons or scope in response to pilot findings. Routine implementation repairs that restore the specified behavior are permitted and logged. A factual novelty problem, invalid comparator, information leak, unavailable required data or material infeasibility that cannot be repaired within the specification is reported as a blocker; it does not authorize a substitute protocol. Any scientific amendment requires a new explicit user instruction and a separately recorded amendment, preserving this lock.

The preliminary package may remain useful even if incomplete. No empirical outcome guarantees publication. The ambition is a clear, consequential question with a feasible route to a defensible conclusion.

### Published baseline context

SmolVLA Table 2 reports 92% LIBERO-Goal success for the 0.45B model. Its simulation recipe uses 100,000 updates, batch 64, and the table identifies initialization from the VLM without robot-data pretraining [R8]. Those conditions differ from our ten-demonstration, 5,000-update S pilot. The published number does not establish our low-data ceiling or the performance of our selected base checkpoint.

### Writing and evidence schedule

Develop the complete scientific manuscript in parallel with implementation, using explicit placeholders for unavailable empirical evidence. Aim for a complete scientific draft by 24 November. The hard preliminary-evidence cutoff is 28 November 2026 at 18:00 Europe/Paris (17:00 UTC). Use completed, audited evidence available by then; describe valid partial work at its actual scope and unfinished work as planned. No further pilot campaign or favorable-result search is authorized after the cutoff. The eight accelerator-job-hour and twelve execution-hour ceilings still apply, with the earliest applicable limit controlling execution.

Reserve 29 November through 3 December for evidence integration, scientific review, source/claim checks, figures and anonymous submission formatting. Target an internally ready PDF on 3 December, ahead of the official deadline. This is an execution and writing plan, not an automated reminder or a claim that manuscript drafting or experiments have already been completed.

## Sources

[R1] RSS 2027 Call for Papers, accessed 20 November 2026. https://roboticsconference.org/information/cfp/

[R2] RSS 2027 Author FAQ. https://roboticsconference.org/information/author-faq/

[R3] RSS 2027 Reviewer Guidelines. https://roboticsconference.org/information/reviewer-guidelines/

[R4] J. Shi et al. Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs. ICML 2026. https://proceedings.mlr.press/v306/shi26aa.html

[R5] Y. Hu et al. AR-VLA: Autoregressive Action Expert for Vision-Language-Action Models. 2026. https://arxiv.org/html/2603.10126v2

[R6] S. Li et al. PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation. 2026. Abstract-level comparison only. https://arxiv.org/abs/2609.21753

[R7] J. Wang et al. SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation. 2026. https://arxiv.org/abs/2608.09771

[R8] M. Shukor et al. SmolVLA: A vision-language-action model for affordable and efficient robotics. 2025. https://arxiv.org/html/2506.01844v1

[R9] Official LeRobot implementation and configuration, moving source inspected 20 November 2026; freeze actual revisions before execution. https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla

[R10] Official LeRobot LIBERO interface. https://huggingface.co/docs/lerobot/main/en/libero

[R11] Robosuite controllers. Documentation version may differ from installed LIBERO dependency; installed pinned code is authoritative for command semantics. https://robosuite.ai/docs/modules/controllers.html

[R12] Official Panda gripper implementation, inspected through GitHub. Its command formatting updates persistent internal action state; it must not be confused with measured finger displacement. https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/grippers/panda_gripper.py
