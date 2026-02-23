.PHONY: paper clean

paper: paper.pdf

paper.pdf: paper.md paper.bib
	pandoc paper.md --from markdown+tex_math_single_backslash --citeproc --pdf-engine=pdflatex -o paper.pdf

clean:
	rm -f paper.pdf
