# `make` = draft build of main.pdf. `make submit` = submission build: any leftover
# \TODO, \UNMEASURED or \com is a hard LaTeX error. `make tldr` prints the outline
# (every %TL;DR line in reading order).

LATEX  = pdflatex -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX = bibtex
SECS   = sec/teaser.tex sec/0_abstract.tex sec/1_intro.tex sec/2_related.tex \
         sec/3_method.tex sec/4_setup.tex sec/5_results.tex sec/6_conclusion.tex
SRC    = preamble.tex cvpr.sty $(SECS) references.bib references_new.bib

.PHONY: all submit clean tldr
all: main.pdf

PRE =
define build
	$(LATEX) "$(PRE)\input{main}" >/dev/null || (grep -A3 '^!\|:[0-9]*: ' main.log | head -30; false)
endef

main.pdf: main.tex $(SRC)
	$(call build)
	$(BIBTEX) main >/dev/null || true
	$(call build)
	$(call build)
	@grep -E "undefined|Rerun" main.log || echo "No undefined references."

submit:
	$(MAKE) clean
	$(MAKE) all PRE='\def\submission{}'

tldr:
	@for f in $(SECS); do echo "== $$f"; grep -h '%TL;DR' $$f | sed 's/^ *%TL;DR:* */  - /'; done

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.brf *.fls *.fdb_latexmk main.pdf
