# RSS 2027 Stage 1: Scientific writing audit and editorial revision v5.3

## Decision

The original v5.2 had a strong scientific question and valid formal experimental comparisons, but the preliminary section and Stage 2 methodology sometimes read as an implementation report. Revision v5.3 is an editorial change; it does not revise the locked experimental design.

## Specific improvements

1. **Argument before procedure.** The abstract and introduction now expose the supervision choice, the proposed transfer mechanism, and the scientific consequence before naming training mechanics.
2. **A concrete first-page illustration.** New Fig. 1 depicts commanded versus observed motion from the same action sequence. The example is schematic rather than an empirical measurement. It introduces no undefined mathematical symbols. The existing transfer and acquisition figures become Figs. 2 and 3.
3. **Conceptual related work.** Prior work is presented as action pretraining versus predictive sensorimotor representation learning, followed by the precise difference in information flow of the present approach.
4. **Terminology.** The manuscript consistently distinguishes the *measured effect* $e_t^D$, *nominal effect* $e_t^K$, and the action-prior regime $P$. These are pretraining regimes, not sequential pipeline components. Conditions $S$ and $E$ are introduced in the introduction before the formalization invokes them.
5. **Why each method decision follows.** The shared native flow-matching objective and parameter transfer are motivated by their role in isolating supervision, rather than listed as implementation features.
6. **Evidence organized by propositions.** The preliminary section now follows the questions: are nominal and measured effects distinguishable, is action generation responsive to measured effect targets, and can downstream task training run in the proposed setting?
7. **Scientific treatment of planned outcomes.** Stage 2 asks whether the learned behavior transfers and when any difference is practically meaningful. Acquisition diagnostics, paired confidence intervals and outcome interpretations remain, but are integrated into the causal argument rather than presented as a work-log.
8. **Concise figure captions.** Captions state the comparison, the necessary controls, and the main inference without becoming technical appendices.
9. **Conclusion.** The conclusion is now a specific three-sentence scientific synthesis rather than a recap of experiment administration.

## Scientific invariants retained

- Same title, model, two primary contrasts $C_1$ and $C_2$, hypotheses $H_1$, $H_2$, and five pretraining/reference regimes $P$, $K$, $D$, $S$, $E$.
- Same four displayed equations, native flow-matching objective, action chunk length $H=16$, six-dimensional endpoint effects, and deployment interface.
- Same central preliminary facts: 100 episodes, 20,000 transitions; 56.2 mm and 16.2 degrees nominal versus 13.1 mm and 3.8 degrees measured; 99.0% and 96.5% residual predictability; no movable-object contacts in that collection; D acquisition 89.1% and 78.1% with matched/masked/alternate controls; S at 158/200; common measured-goal comparison with the analytic 29.7% reference and K's native 1/64 and 0/64 outcome.
- Same Stage 2 data, training and test commitments: 250,000 transitions, 10 and 40 demonstrations per task, eight paired seeds, 50 resets per task, 5,000 intermediate and 10,000 downstream updates, and five percentage points as the practical threshold.
- No new empirical results, training arms, hypotheses, or inferential claims.

## Remaining production notes

- The attached PDF preview was compiled using the available system IEEEtran v1.8b. The GitHub paper branch contains the official RSS IEEEtran v1.7a. Rebuild with that official class before submission; after the additional first-page illustration, verify the exact official layout and page/reference split.
- The current scientific content fits four content pages plus one references page in the preview, with no unresolved references or overfull boxes.
- All three figure sources are editable TikZ. Grayscale and scale should be checked after a PowerPoint redraw. The source branch remains anonymous; do not link it in the RSS paper.