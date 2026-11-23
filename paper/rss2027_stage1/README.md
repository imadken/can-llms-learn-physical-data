# RSS 2027 Stage 1 anonymous manuscript

**Title:** What Transfers from Effect-Conditioned Action Pretraining to Vision-Language-Action Policies?

This folder contains the paper only. Experimental code, datasets and private audit artifacts remain outside it.

## Build with the official RSS class

The *unmodified* official RSS LaTeX kit's IEEEtran.cls **V1.7a** is committed under `template/`. It was extracted from the RSS conference's official archive at https://roboticsconference.org/docs/paper-template-latex.tar.gz. The archive SHA256 is `ffc8b8053bd218ba9be4c93a569de4c2f36c7f70ae8d2245c747611bed697271`. The exact class SHA256 is `d63d4bf27f29603a733d06e27ba5fbb40488acf9d1d696354a88b886758f70a7`.

From this directory:

```bash
make template
make
make check
```

`make` uses `TEXINPUTS` to load the committed official class, *not* your TeX installation's potentially newer IEEEtran class. No Internet access is needed if the committed class is present. Do not modify the class, paper size, font sizes, margins, or spacing.

The current submission candidate was preview-compiled with the system IEEEtran **V1.8b** in the assistant's environment, which did not have the official V1.7a installed locally. **You must rebuild once with `make` in this directory and verify the resulting page count and layout before submission.** The preview PDF is not a verified official-class build.

## Editing

- `main.tex`: full anonymous draft, entry point for compilation.
- `figures/transfer.tex`: editable vector thesis/transfer figure.
- `figures/acquisition.tex`: editable vector pilot figure.
- `references.bib`: citation database used by `plainnat.bst`; the manuscript uses the RSS kit's required `natbib` + `plainnat` workflow. BibTeX is invoked automatically by `latexmk`.
- `FIGURE_REDRAW_GUIDE.md`: precise scientific specification of both figures.
- `SUBMISSION_CHECKLIST.md`: production and anonymity checks.

The figure sketch PowerPoint is delivered separately in the ChatGPT conversation and can be copied into this directory if desired. The submission uses editable LaTeX figures until they are explicitly replaced.

**Privacy:** this GitHub repository is public. The branch's anonymous paper source should not be linked from the submission; consider making the repository private before review to protect double-blind anonymity.
