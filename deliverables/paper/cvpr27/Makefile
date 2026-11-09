# CVPR 2027 build. `make` = draft build of main.pdf + supp.pdf.
# `make submit` = submission build (\draftfalse): any leftover \TODO, \UNMEASURED
# or \com is a hard LaTeX error.
# main and supp cross-reference each other (xr-hyper), so each is built twice
# around the other's .aux.

LATEX  = pdflatex -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX = bibtex
SRC    = preamble.tex cvpr.sty $(wildcard sec/*.tex) references.bib references_new.bib

.PHONY: all submit clean
all: main.pdf supp.pdf

PRE    =
define build
	$(LATEX) -jobname=$(1) "$(PRE)\input{$(1)}" >/dev/null || (grep -A3 '^!\|:[0-9]*: ' $(1).log | head -30; false)
endef

main.pdf supp.pdf: main.tex supp.tex $(SRC)
	$(call build,supp)
	$(call build,main)
	$(BIBTEX) supp >/dev/null || true
	$(BIBTEX) main >/dev/null || true
	$(call build,supp)
	$(call build,main)
	$(call build,supp)
	$(call build,main)
	@grep -E "undefined|Rerun" main.log supp.log || echo "No undefined references."

submit:
	$(MAKE) clean
	$(MAKE) all PRE='\def\submission{}'

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.brf *.fls *.fdb_latexmk main.pdf supp.pdf
