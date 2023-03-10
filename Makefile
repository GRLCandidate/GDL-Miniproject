PANDOC_CMD = pandoc
PANDOC_OPTS = --citeproc --csl chicago-author-date.csl --template ./template.latex
IN_FILE = main.md
OUT_FILE = 1045966-geometric_deep_learning-miniproject.pdf
BBT_URL = http://127.0.0.1:23119/better-bibtex/export/collection?/1/IWJNI9X3.biblatex

# images = img/1-line.png img/2c-g.png img/2e-wl.png img/1-clusters.png

all: paper

watch:
	echo template.latex Makefile ${IN_FILE} $(wildcard img/*) \
		| sed -e "s/ /\n/g" \
		| entr make paper

clean:
	rm $OUT_FILE

paper: references # $(images)
	${PANDOC_CMD} ${PANDOC_OPTS} -o ${OUT_FILE} ${IN_FILE}

references:
	@wget -O references.biblatex "${BBT_URL}" | echo "WARNING: Could not fetch references. Make sure Zotero with BBT is running." >&2

img/%.gv.png: gv/%.dot
	dot -Tpng $^ > $@

.PHONY: all watch clean paper references
