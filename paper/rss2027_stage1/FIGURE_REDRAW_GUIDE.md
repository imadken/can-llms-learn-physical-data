# RSS 2027 v5.5 figure guide

The three figures serve different scientific purposes: introduce the information difference, explain the transfer mechanism and show numerical acquisition. Keep terminology consistent with the manuscript: action prior P, nominal-effect conditioning K and measured-effect conditioning D.

## Figure 1: One interaction, two effect descriptions

**Source:** `figures/motion_example.tex`.

The first-page figure now shows audited empirical means rather than a cartoon: nominal translation magnitude 56.2 mm and measured magnitude 13.1 mm. The population is 200 distinct start/chunk pairs from 100 restored snapshots in ten held-out pilot episodes, each replayed twice. These are scalar magnitudes; the bars do not reconstruct trajectories or depict two mean endpoint positions. The figure makes no contact-effect claim. No movable-object contact occurred in this pilot.

Keep the labels Nominal and Measured, the common linear millimeter axis, and the numeric values beside the bars. Avoid perspective or decorative lengths that distort their ratio. Retain the caption's reference to the preliminary-evidence section for the sampling context.

## Figure 2: What the action expert learns and retains

**Publication source:** `figures/transfer.pdf`. **Editable sources:** `figures/transfer.svg` and `figures/transfer_editable.pptx`.

The architecture keeps two connected phases. Numerical learning supplies a robot observation, an action-training target and one of three alternative conditions. Task learning supplies images, a language instruction and robot state to the retained action generator. The central visual message is the distinction between the temporary effect-conditioning interface and the action-generation parameters that carry across phases.

The nominal branch uses the initial observation and command chunk. Only the measured branch uses the subsequent observed outcome. P has a neutral condition. Do not draw future observations entering the native state input or all three regimes. Depict the action-training target through its flow/noise training pathway, rather than as a clean action copied directly to the output.

The vision-language backbone stays frozen. The native state projector is frozen during numerical pretraining and trainable during task fine-tuning. The action generator is trainable in both phases. The temporary effect projector is discarded; no effect token or additional effect module remains at deployment. Keep the transfer arrow on retained parameters, not on a claimed learned dynamics model.

Open the PPTX to edit labels and diagram elements. Use the SVG when working in a vector editor. The manuscript PDF outlines Figure 2 lettering for portable vector rendering, so editing that exported lettering is less convenient than editing the source. After any revision, re-export the publication PDF and inspect it at the final two-column paper width. Icons must clarify inputs, conditions or transfer; avoid decoration that competes with these paths.

## Figure 3: Numerical goals influence generated actions

**Source:** `figures/acquisition.tex`.

Panel (a) reports D's original-target attainment for matched inputs (89.1%, 78.1%), masked inputs (12.5%, 20.3%) and mismatched inputs (6.3%, 1.6%). The mismatched-input test is still scored against the **original target**. Low original-target attainment shows sensitivity to changing the supplied condition; it does not measure successful attainment of the alternative goal.

Panel (b) uses a common measured-goal test: P (14.1%, 20.3%), K (51.6%, 50.0%) and D (89.1%, 78.1%). This changes K's conditioning distribution from its nominal training goals. It is not the downstream task comparison. The 29.7% dashed reference is the analytic zero-displacement reference, not an executed zero-action policy. Each seed has 64 targets; the joint tolerance is 10 mm translation and 5 degrees rotation.

Keep the same seed encodings in both panels, a zero-based percentage axis and labels that remain distinguishable in grayscale. The accompanying text retains K's separate native nominal-goal result, 1/64 and 0/64 at the strict joint tolerance.

## Final checks

Every figure is referenced in the paper. Check label size at paper scale, caption consistency, vector rendering, grayscale legibility and absence of overlaps. Do not replace these evidence plots with generated imagery or add inferred measurements. The v5.5 release passed these checks, including the outlined Figure 2 export and the complete official-class manuscript PDF.
