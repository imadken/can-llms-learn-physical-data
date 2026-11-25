# RSS 2027 Stage 1 scientific writing audit, v5.6

## Scope and narrative

This is an editorial and formalization revision of the locked research plan. No new training, pilot result, comparison regime or primary hypothesis is introduced. The manuscript remains a Stage 1 research paper: the problem and method are specified, existing evidence supports prerequisites, and the decisive downstream evaluation is planned.

The narrative follows one question: which information from a robot interaction helps subsequent task learning? An action prior uses the actions; nominal conditioning adds their command-derived effect; measured conditioning uses their observed effect. The same action-generation parameters then transfer to ordinary image-language-state conditioning. Numerical acquisition and downstream transfer are separate scientific questions.

## Decisions in response to the author review

- Remove the model name from the abstract and general methodology. Identify the public SmolVLA base checkpoint in the experimental setting for reproducibility.
- Preserve the introduction's motivating example and comparative story. Replace its schematic Figure 1 with audited replay means.
- Generalize the formal problem before instantiating it with six-dimensional end-effector motion. This expresses scope without claiming tested transfer beyond manipulation.
- Define generic demonstration budget `b` and contrasts `C1(b)` and `C2(b)` in the formulation. Specify the primary ten-demonstration setting in the planned evaluation.
- Explain retained parameters as the hypothesized carrier of conditional action-generation behavior. Removing the temporary goal token does not itself establish transfer.
- Keep the numerical corpus and the training schedule in the planned evaluation. Preserve E's exact compute-matched task-training comparison, while reducing the proposed experimental scale for Stage 2 feasibility.
- Present the outcome table around interpretable comparisons, including nonpositive C1, unestablished acquisition and unresolved intervals. Avoid treating an interval crossing zero as equivalence.
- Keep the material pilot facts: no movable-object contact, K's native 1/64 and 0/64 strict attainment, and the common measured-goal distribution shift. Explain their role directly, without speculative failure narratives.
- Keep the calibration and intermediate-dose diagnostics as possible Extension Charter studies. They are not newly authorized training arms.

## Mathematical and notation checks

Observations, actions, chunk horizon and dimensions are defined before use. The general effect map `phi` summarizes an executed change; nominal map `G` uses the initial observation and actions without future measurements. The concrete instance specifies world-frame translation and an initial-end-effector-frame relative rotation. SO(3) is defined as the proper three-dimensional rotation matrices; transpose and the principal rotation-vector logarithm are explained, with its angle branch stated.

The formulation distinguishes physical action chunks from normalized and possibly padded model targets. The flow target `epsilon - A_bar` is consistent with interpolation from data at time zero to noise at time one; sampling integrates in the reverse time direction. The squared Frobenius norm covers physical coordinates and chunk steps, with the normalization stated. Projector parameters, action-generation parameters, numerical conditions and masking are defined. The experimental setting specifies the native time distribution and normalization floors.

The native state projector is frozen during numerical pretraining and trainable during task fine-tuning. P, K and D are alternatives, not sequential stages. D replaces K's condition rather than appending a residual. The pretrained VLA checkpoint is shared; the method does not claim that the model has never encountered demonstrations.

## Retained analysis protocol

The main manuscript now emphasizes effect sizes, paired uncertainty and interpretable outcomes. The following numerical commitments remain in the research protocol, rather than being silently discarded by the style revision.

- Independent unit: one of six paired pipeline seeds. Task means receive equal weight; evaluation resets do not become independent training replicates.
- Primary estimates: `C1(10) = J10(K) - J10(P)` and `C2(10) = J10(D) - J10(K)`.
- Each two-sided paired Student interval has 97.5% marginal coverage. Bonferroni gives at least 95% simultaneous coverage for the two primary contrasts under the paired-seed model.
- With six seeds, the half-width is `t(5, 0.9875) * s_d / sqrt(6)`, approximately `1.291 * s_d`, where `s_d` is the sample standard deviation of paired seed differences.
- A positive difference has its interval wholly above zero. This directional statement is distinct from the prespecified practically meaningful gain of five percentage points.
- The retained practical margin is `delta = 0.05`: a practically meaningful gain requires the lower interval bound above delta; a gain of that size is excluded when the upper bound is below delta; practical equivalence requires the entire interval within `[-delta, delta]`. Other cases remain unresolved for the corresponding question.
- Consequently there is no universal required 9 to 10 point gap. The practical-gain criterion is the estimated gap exceeding `0.05 + 1.291 * s_d`. Equivalence requires `abs(estimated gap) + 1.291 * s_d < 0.05`. At a five-point paired standard deviation, the half-width is about 6.45 percentage points, so equivalence cannot be established even when the estimated difference is zero.

The formal practical margin is not repeated throughout the Stage 1 narrative. It remains recorded here before downstream outcomes are observed. Supported positive effects in the manuscript must not be relabeled as practically meaningful effects unless the retained margin is met.

## Evidence and resource interpretation

The existing pilot contains 20,000 transitions from 100 episodes in one scene. Figure 1 reports the means for 200 distinct start/chunk pairs from 100 restored snapshots in ten held-out episodes, with two replays per pair. These are mean translation magnitudes, not a reconstructed path or a mean endpoint vector. The action-acquisition test uses two seeds and 64 targets per seed; the standard task-training check uses one seed and 200 rollouts. No downstream P/K/D/E transfer result is reported.

The Stage 1 proposed scale is now a shared numerical corpus targeting 100,000 transitions, six paired seeds, 18 intermediate fits, 60 multitask fine-tunes and 30 fixed evaluation resets per task (18,000 final-policy rollouts). The five-regime, two-budget comparisons, fixed task suite and intermediate/final training schedules are unchanged. The scientific analysis is prespecified; any additional validation requested by the RSS Extension Charter will be discussed after Stage 1 review. No hardware-specific or projected runtime claim is made in the manuscript.

No additional one-seed run at the Stage 2 schedule was performed for this revision. Such a run could inform schedule feasibility and native K acquisition but would not settle the benchmark ceiling or training-seed variability. The reported S pilot establishes an operational evaluation setting, not convergence or a population-level headroom estimate.

## Reference verification

All thirteen cited works were checked against primary records or author-maintained sources. The bibliography retains its citation keys. The substantive metadata corrections are OpenVLA's PMLR publication year, 2025, and AR-VLA's author name, Danda Pani Paudel. AR-VLA's DOI is added. Sources appear below so verification does not depend on internal tool references.

| Work | Verified source and decision |
| --- | --- |
| pi0 | [RSS 2025 proceedings](https://www.roboticsproceedings.org/rss21/p010.html): existing DOI and title correct. |
| SmolVLA | [arXiv](https://arxiv.org/abs/2506.01844): title, fourteen authors and 2025 preprint correct. |
| OpenVLA | [PMLR](https://proceedings.mlr.press/v270/kim25c.html): volume 270, pages 2679 to 2713, publication year 2025; the conference event was in 2024. |
| MCR | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cb9e74b26c174b60e94e14c73646a55d-Abstract-Conference.html): title, authors and venue correct. |
| TAP | [PMLR](https://proceedings.mlr.press/v306/shi26aa.html): ICML 2026, volume 306, pages 112301 to 112319 confirmed. |
| AR-VLA | [RSS 2026 proceedings](https://www.roboticsproceedings.org/rss22/p085.html): final author's full name and DOI `.085` confirmed. |
| PSR | [Full methods text](https://arxiv.org/html/2609.21753v1): future sensorimotor prediction, terminal displacement supervision and retained predictive hierarchy verified. |
| SLIM | [arXiv record](https://arxiv.org/abs/2608.09771) and [official implementation](https://github.com/kzz1031/SLIM): metadata and combined inverse/forward prediction verified; full paper not inspected. |
| Flow Matching | [Author paper](https://arxiv.org/abs/2210.02747) and indexed [ICLR PDF](https://openreview.net/pdf?id=PqvMRDCJT9t): original method, authors and ICLR 2023 venue verified. |
| LIBERO | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html): metadata correct. |
| Pose-VLA | [RSS 2026 proceedings](https://www.roboticsproceedings.org/rss22/p145.html): title, authors and DOI `.145` correct. |
| Tune to Learn | [RSS 2026 proceedings](https://www.roboticsproceedings.org/rss22/p139.html): title, authors and DOI `.139` correct. |
| APT | [Author project page](https://xukechun.github.io/papers/APT/) and [author repository](https://github.com/xukechun/APT): authors explicitly identify CoRL 2026 and provide conference BibTeX. No proceedings volume or pages are inferred. |

Measured-effect conditioning is positioned as goal-conditioned inverse-dynamics learning. Novelty resides in the controlled choice of supervision, not in claiming the first use of future outcomes. PSR and SLIM have distinct objectives and are described separately. In particular, PSR already uses terminal displacement supervision and SLIM already includes inverse dynamics.

## Production status

The unmodified official IEEEtran V1.7a class is now used, with a five-content-page plus one-reference-page build. The final exported figures and every PDF page were visually inspected. The build has resolved references, no overfull boxes, anonymous metadata, and only embedded Type 1 fonts. The reference list occupies exactly one page. The previous v1.8b-only preview limitation is resolved by the official-class build, not by changing the class or paper dimensions.
