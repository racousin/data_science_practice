# Courseware build — Markdown source of truth -> PPTX decks + ML-Arena course.
#
# Dependencies are pulled per-invocation by `uv run --with`, so there is no
# environment to create or keep in sync.

COURSE      ?= python-ai-engineering
CONTENT     := content/$(COURSE)
OUT         := build
BASE_URL    ?= https://ml-arena.com
SDK_PATH    ?= ../../../mlarena-sdk

PY_SLIDES   := uv run --quiet --with python-pptx --with pyyaml --with matplotlib python
PY_PUBLISH  := uv run --quiet --with pyyaml --with requests python

.PHONY: help slides pdf publish publish-dry harvest clean

help:
	@echo "make slides       — build one PPTX per session into $(OUT)/slides"
	@echo "make pdf          — slides, then convert to PDF (needs LibreOffice)"
	@echo "make publish-dry  — show what would change on $(BASE_URL)"
	@echo "make publish      — sync content to ML-Arena (needs MLARENA_API_KEY)"
	@echo "make harvest      — re-extract Markdown from the React site into _harvest/"
	@echo "make clean        — remove $(OUT)/"
	@echo
	@echo "Variables: COURSE=$(COURSE)  BASE_URL=$(BASE_URL)"

# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #

slides:
	$(PY_SLIDES) tools/build_slides.py $(CONTENT) --out $(OUT)/slides --strict-assets

# One session only:  make slides-one MODULE=s2-agentic-coding
slides-one:
	$(PY_SLIDES) tools/build_slides.py $(CONTENT) --out $(OUT)/slides \
		--module $(MODULE) --strict-assets

SOFFICE := /Applications/LibreOffice.app/Contents/MacOS/soffice

pdf: slides
	@test -x $(SOFFICE) || { echo "LibreOffice not found at $(SOFFICE)"; exit 1; }
	@mkdir -p $(OUT)/pdf
	$(SOFFICE) --headless --convert-to pdf --outdir $(OUT)/pdf $(OUT)/slides/*.pptx
	@echo "PDFs in $(OUT)/pdf"

# --------------------------------------------------------------------------- #
# ML-Arena
# --------------------------------------------------------------------------- #

publish-dry:
	$(PY_PUBLISH) tools/publish_mlarena.py $(CONTENT) \
		--base-url $(BASE_URL) --sdk-path $(SDK_PATH) --dry-run

publish:
	@test -n "$$MLARENA_API_KEY" || { \
		echo "MLARENA_API_KEY is not set."; \
		echo "Mint a teacher-scope key (mlk_teacher_...) on your ML-Arena Profile page."; \
		exit 1; }
	$(PY_PUBLISH) tools/publish_mlarena.py $(CONTENT) \
		--base-url $(BASE_URL) --sdk-path $(SDK_PATH)

# --------------------------------------------------------------------------- #
# Migration aid
# --------------------------------------------------------------------------- #

WEBSITE := ../website/src/pages

harvest:
	@for m in data-science-practice/module1 data-science-practice/module2 \
	          data-science-practice/module3 data-science-practice/module4 \
	          data-science-practice/module5 data-science-practice/module6 \
	          data-science-practice/module7 data-science-practice/module8 \
	          data-science-practice/module9 \
	          python-deep-learning/module1 python-deep-learning/module2 \
	          python-deep-learning/module3 python-deep-learning/module4; do \
		out="_harvest/$$(echo $$m | tr '/' '-')"; \
		[ -d "$(WEBSITE)/$$m" ] && python3 tools/harvest_website.py "$(WEBSITE)/$$m" --out "$$out" 2>/dev/null; \
	done
	@echo "harvested to _harvest/ (gitignored) — raw material, not deliverables"

clean:
	rm -rf $(OUT)
