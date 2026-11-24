# Scientific figure redraw guide (PowerPoint)

The accompanying `figures_powerpoint/RSS2027_Editable_Figure_Sketches.pptx` contains **native PowerPoint shapes, lines and editable text**, not flattened screenshots. It is a sketch and not part of the anonymous submission unless you export it and replace the current vector figure sources.

## Figure 1. Supervision choice and the retained action generator

**Scientific question:** Which numerical supervision, rather than which architecture, creates reusable action-generation parameters?

1. Draw two horizontally aligned regions, **(a) Numerical intermediate training** and **(b) Multimodal task fine-tuning**.
2. In (a), start from the **same robot observation** `r_t` and **action chunk** `A_t`. Split into three explicit conditioning cases: `P` = neutral effect input, `K` = nominal endpoint effect `e_t^K`, `D` = measured endpoint effect `e_t^D`. These are alternatives, not three sequential stages.
3. Let the three alternatives join the **same temporary effect projector** `h_η`, feeding the **native flow-matching action expert** `v_θ`. The training target is the action chunk `A_t` under the native flow objective, not a forward-dynamics prediction head.
4. Across the two halves, show **retained action-generation parameters** transferred; **discard the temporary projector and effect token**. It is the parameters that transfer, not measured endpoints and not an additional module deployed at test time.
5. In (b), show **image and instruction** passing through the frozen vision-language encoder; **current robot state** through the native state projector. Both condition the retained expert, which outputs an action chunk. There is **no measured future endpoint** at deployment.
6. Add one miniature explanatory inset: for a fixed action chunk, the commanded ideal endpoint and the observed endpoint can differ. The inset is schematic, not a drawn-to-scale empirical measurement. Consistent labels: K = command-derived, D = observed.
7. Avoid imagery suggesting K and D are different controllers or robot embodiments. The action expert, robot, action format, numerical windows, training objective, and deployed policy are controlled.

**Suggested paper caption:** “Controlled numerical supervision and transfer. Action prior (P), nominal endpoint (K), and measured endpoint (D) conditions share trajectories and the native action generator. Only temporary numerical conditioning differs. The projector is discarded and the trained action-generating weights initialize ordinary multimodal fine-tuning.”

## Figure 2. Numerical goal acquisition

**Scientific question:** Does the numerical conditioning objective produce behavior responsive to the supplied endpoint, and how do the defined training regimes behave on a common goal test?

1. Two plots with a shared percent scale **0–100%**, jointly requiring **10 mm translation and 5° rotation** accuracy. Plot **two distinct seed markers**, not a pooled bar.
2. Panel (a) D acquisition on measured targets: matched goals **89.1 / 78.1%**, masked **12.5 / 20.3%**, alternative target **6.3 / 1.6%**, seeds 101/202.
3. Panel (b), common **measured** targets: P **14.1 / 20.3%**, K **51.6 / 50.0%**, D **89.1 / 78.1%**, seeds 101/202. State next to panel (b) that **K was trained on nominal targets**, so this is a distribution-shifted test for K.
4. Include **29.7% zero-displacement reference** as a dashed line in panel (b), clearly labeled **analytic, not executed**. Do not include it in panel (a).
5. Caption includes **64 targets per seed from four source episodes**, and “exploratory acquisition, not a downstream VLA transfer comparison.”
6. Use identical glyphs for both seeds across panels, grayscale-safe (filled dot / open circle). Do not interpret panel (b) as a comparative estimate of task success after pretraining.

**Suggested paper caption:** “Exploratory numerical goal acquisition. (a) Matched, masked and alternative effect conditioning for D. (b) P, K and D evaluated on common measured goals; K was pretrained on nominal goals. Symbols denote two seeds (64 targets each, four episodes). Dashed line: analytic zero-displacement reference. Joint accuracy requires 10 mm and 5°; no downstream transfer is evaluated.”

## PowerPoint and export details

- Use a **wide two-panel layout** suitable for a full-width `figure*` in a two-column paper.
- Keep every label in the figure consistent with manuscript symbols; do not switch between `G` and `g`, `A_t` and `a_t`, or `e^D` and `e_t^D` without defining the difference.
- Use at least 9 pt text in the final *printed* figure. The sketches are larger than their final paper scale, so simplify rather than shrinking all labels.
- Avoid heavy gradients and stock robot pictures. One action→goal illustration with simple arrows communicates the hypothesis better.
- Prefer vector output **PDF/EPS** and embedded fonts accepted by RSS. Inspect `pdffonts` after integrating new figures; screenshot images should not replace the main scientific result.
- The current paper includes compact fully editable TikZ figure sources. Replace them only after verifying the redraw against the science and the final page limit.
