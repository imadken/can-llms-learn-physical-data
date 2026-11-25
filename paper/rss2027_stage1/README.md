# RSS 2027 Stage 1 manuscript, revision v5.6

**What Transfers from Effect-Conditioned Action Pretraining to Vision-Language-Action Policies?**

This revision develops the scientific narrative from the supervision question through the method, preliminary evidence and planned transfer study. The five regimes, paired comparisons, training schedules and audited pilot observations remain unchanged. The proposed Stage 2 scale is reduced to a 100,000-transition collection target, six paired seeds and 30 evaluation resets per task. No new training or experimental result is introduced.

## Contents

- `main.tex` and `references.bib`: anonymous Stage 1 manuscript and verified bibliography.
- `figures/motion_example.tex`: empirical nominal/measured translation comparison for Figure 1.
- `figures/transfer.pdf`: publication version of Figure 2, with outlined lettering; `transfer.svg` and `transfer_editable.pptx` provide editable versions.
- `figures/acquisition.tex`: Figure 3, with mismatched goals scored against the original target.
- `SCIENTIFIC_WRITING_AUDIT.md`: editorial decisions, reference verification and retained analysis commitments.
- `FIGURE_REDRAW_GUIDE.md`: figure meanings, edit instructions and interpretation constraints.

The audit and figure guide are author-facing working documents. They are not manuscript supplements. RSS Stage 1 accepts no supplementary material, so the PDF contains the scientific essentials.

## Build

Run `make` in `paper/rss2027_stage1/`, with LaTeX, latexmk, BibTeX, natbib, plainnat, TikZ, PGFPlots and the packages imported by `main.tex` installed. The Makefile selects `template/IEEEtran.cls` explicitly through `TEXINPUTS`. The template script checks the official class and obtains it from the conference archive if absent.

The build uses the unmodified RSS class **IEEEtran 2007/03/05 V1.7a**, SHA-256:

```text
d63d4bf27f29603a733d06e27ba5fbb40488acf9d1d696354a88b886758f70a7
```

The current official-class build has **six pages: five content pages and one reference page**. This replaces the earlier system-class preview. Run `make check` for page and font information; it does not replace visual inspection or reference checks.

## Final production check

The v5.6 release was built with the pinned official class and checked page by page. It has five content pages followed by exactly one reference page, resolved citations and cross-references, no overfull boxes or clipped content, readable figure labels, anonymous PDF metadata, and only embedded Type 1 fonts. Figure 2 lettering is outlined as vector paths in the publication PDF; the PowerPoint and SVG remain editable. Recheck the export after any subsequent edit.

Author walkthroughs, if included outside the paper directory, are teaching material rather than part of the anonymous submission.
