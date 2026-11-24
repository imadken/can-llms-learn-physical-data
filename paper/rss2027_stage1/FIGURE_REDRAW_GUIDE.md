# RSS 2027 v5.3: Scientific figure redraw guide

The paper contains **three visual roles**, rather than treating every figure as an architecture or reporting chart. All figure sources are editable TikZ and can be redrawn as PowerPoint vector shapes.

## Figure 1: An intuitive example on the opening page

**Message:** The same action sequence produces a command-defined nominal motion and a realized measured motion. These are distinct candidate signals for intermediate learning.

Draw a start-point marker at the left. From that marker, show (i) a dashed grayscale arrow to the nominal requested endpoint and (ii) a solid blue/black arrow to the observed endpoint. Label the arrows *Commanded motion* and *Observed motion*. Put the sentence *One action sequence, two motion descriptions* above the panel. The drawing is illustrative, not an empirical trajectory; avoid inserting fabricated displacement values. Keep the caption to one explanatory sentence.

## Figure 2: What is transferred and what is removed

**Message:** Three alternative numerical supervision regimes train a common native action expert, but only its action-generation parameters transfer to the standard multimodal task policy.

Use two regions. Left: numerical observations and action chunks → three alternative endpoint inputs P, K, D → temporary effect projector → native action expert. Distinguish this as alternatives, not sequential stages. Right: images and instruction → frozen vision-language encoder; robot state → native state projector; both condition the retained action expert → action chunk. A single transfer arrow should link the trained action expert to itself across the panels. Mark the temporary projector as discarded. Draw neither a physical parameter predictor nor an additional deployed module. Use labels and symbols consistent with Section III and Eq. (4).

## Figure 3: What the acquisition evidence establishes

**Message:** Measured-effect pretraining produces goal-sensitive actions in the numerical pilot; the plot is not a downstream VLA transfer ranking.

Panel (a): D's joint endpoint attainment on matched (89.1%, 78.1%), masked (12.5%, 20.3%), and alternative (6.3%, 1.6%) goals for seeds 101 and 202.

Panel (b): shared measured goals: P (14.1%, 20.3%), K (51.6%, 50.0%), and D (89.1%, 78.1%). Show the 29.7% analytic zero-displacement reference as a dashed line, marked **not an executed policy**. Label K's input as out of its nominal-goal training distribution. Use the same seed encodings across both panels. State that each seed covers 64 goals and that joint attainment requires 10 mm translation and 5 degrees rotation accuracy.

### Presentation checks

Keep the figures vector-editable; ensure that grayscale printing retains dashed/solid differentiation. Test font size at the final paper scale. Every figure and table is referenced in the manuscript text. Avoid code names, file paths, GPU details, and training logs in figure labels or captions.