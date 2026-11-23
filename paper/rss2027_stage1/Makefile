# Requires official RSS IEEEtran.cls. Template is downloaded from RSS and left unmodified.
.PHONY: all template clean check
all: template
	TEXINPUTS="$(CURDIR)/template//:" latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

template:
	bash template/get_official_template.sh

check:
	@test -f template/IEEEtran.cls || { echo 'Run make template'; exit 1; }
	@grep -q '2007/03/05 version V1.7a' template/IEEEtran.cls
	@pdfinfo main.pdf | grep 'Pages:'
	@pdffonts main.pdf

clean:
	latexmk -c -pdf main.tex
