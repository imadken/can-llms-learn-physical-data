# RSS 2027 Stage 1 paper

**Title:** What Transfers from Effect-Conditioned Action Pretraining to Vision-Language-Action Policies?

This directory contains only the anonymous submission manuscript and its figure sources. Experimental code, checkpoints, datasets, audits and internal research notes stay outside the paper directory. `main.tex` is the compile entry point.

## Official template, exact verified version

The RSS author kit is provided at <https://roboticsconference.org/docs/paper-template-latex.tar.gz> and pins the unmodified **IEEEtran.cls 2007/03/05 V1.7a**. The SHA256 of the template archive verified against a public RSS source is:

`ffc8b8053bd218ba9be4c93a569de4c2f36c7f70ae8d2245c747611bed697271`

Run from this folder:

```bash
make template        # one-time fetch and SHA check of the official kit
make                 # builds main.pdf with that class
make check           # checks class, page count, PDF fonts
```

Requires `curl`, `tar`, `sha256sum`, `latexmk`, `pdflatex`, TikZ, PGFPlots, natbib and pdffonts. If the official archive changes, the script fails deliberately until its update is inspected; it will not silently compile using a different template. **Do not change the class, font sizes, margins or line spacing.**

The source was also locally compiled as a preview with the system IEEEtran V1.8b because the tool environment cannot directly fetch the official binary. That preview does not replace a final compilation using V1.7a. Verify the PDF produced by `make` before OpenReview upload.

## Files

- `main.tex`: anonymous 5+1 manuscript.
- `references.bib`: editable reference metadata; current `main.tex` contains the numbered reference list needed for stable one-page references.
- `figures/transfer.tex`: transfer architecture, vector TikZ.
- `figures/acquisition.tex`: exploratory acquisition evidence, vector PGFPlots.
- `figures_powerpoint/RSS2027_Editable_Figure_Sketches.pptx`: editable PowerPoint redraw sketches.
- `FIGURE_REDRAW_GUIDE.md`: exact symbols, experimental values, and figure design logic.
- `template/get_official_template.sh`: unchanged official-class acquisition.

## Submission checks

1. `main.pdf` contains up to **5 content pages and exactly one references page**, no external supplementary material, and no page numbers.
2. Recheck anonymization including PDF metadata, author list, source paths, and public repository links.
3. Confirm all citations and results and test grayscale figure readability.
4. Confirm no substantially similar paper is simultaneously under review elsewhere.
5. The GitHub repository is currently **public**. Do not link its paper source in the blinded submission; hosting a uniquely titled draft publicly may facilitate author identification.
