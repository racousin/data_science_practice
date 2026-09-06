# Courseware build — Markdown source of truth -> PPTX decks + ML-Arena course.
#
# Dependencies are pulled per-invocation by `uv run --with`, so there is no
# environment to create or keep in sync.

COURSE      ?= python-ai-engineering
CONTENT     := content/$(COURSE)
OUT         := build
BASE_URL    ?= https://ml-arena.com
# NO_MEDIA=1 publishes lesson text only (see publish_mlarena.py --skip-media).
NO_MEDIA    ?=
SDK_PATH    ?= ../../../mlarena-sdk

PY_SLIDES   := uv run --quiet --with python-pptx --with pyyaml --with matplotlib python
PY_PUBLISH  := uv run --quiet --with pyyaml --with requests python

.PHONY: help slides pdf publish publish-dry harvest clean \
        competitions competitions-status competitions-publish competitions-attach

help:
	@echo "make slides       — build one PPTX per session into $(OUT)/slides"
	@echo "make check-slides — list slides whose content overflows the slide"
	@echo "make pdf          — slides, then convert to PDF (needs LibreOffice)"
	@echo "make publish-dry  — show what would change on $(BASE_URL)"
	@echo "make publish      — sync content to ML-Arena (needs MLARENA_API_KEY)"
	@echo "                    add NO_MEDIA=1 to publish text without images"
	@echo "make harvest      — re-extract Markdown from the React site into _harvest/"
	@echo "make clean        — remove $(OUT)/"
	@echo
	@echo "make competitions          — build/benchmark/start the session competitions"
	@echo "make competitions-status   — show what is live"
	@echo "make competitions-publish  — flip them public"
	@echo "make competitions-attach   — link them to their modules (TEACHER key)"
	@echo
	@echo "Variables: COURSE=$(COURSE)  BASE_URL=$(BASE_URL)"

# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #

slides:
	$(PY_SLIDES) tools/build_slides.py $(CONTENT) --out $(OUT)/slides --strict-assets

# Slides whose content renders off the bottom. The deck builder shrinks the font
# until the content fits and then gives up silently, so this is the only way to
# see it without reading 300 slides. Exits non-zero, so it gates a build.
check-slides:
	$(PY_SLIDES) tools/check_slide_overflow.py $(CONTENT) $(if $(MODULE),--module $(MODULE)) $(if $(VERBOSE),--verbose)

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
		--base-url $(BASE_URL) --sdk-path $(SDK_PATH) $(if $(NO_MEDIA),--skip-media)

# --------------------------------------------------------------------------- #
# Competitions
#
# One per taught session; see competitions/README.md. `build` refuses to start a
# competition whose reference solution does not score its declared benchmark, so
# a green run is evidence the scoring works end to end.
# --------------------------------------------------------------------------- #

PY_COMPS := uv run --quiet --with requests python
COMP_ARGS := --base-url $(BASE_URL) --course $(COURSE)

competitions:
	@test -n "$$MLARENA_API_KEY" || { \
		echo "MLARENA_API_KEY is not set (needs a mlk_creator_... key)."; exit 1; }
	MLARENA_SDK_PATH=$(SDK_PATH) $(PY_COMPS) tools/build_competitions.py build $(COMP_ARGS) $(if $(ONLY),--only $(ONLY))

competitions-status:
	MLARENA_SDK_PATH=$(SDK_PATH) $(PY_COMPS) tools/build_competitions.py status $(COMP_ARGS)

competitions-publish:
	MLARENA_SDK_PATH=$(SDK_PATH) $(PY_COMPS) tools/build_competitions.py publish $(COMP_ARGS)

# attach_competition is an /api/teacher/* route and API-key auth requires the
# scope to match exactly — a creator key is rejected here.
competitions-attach:
	@test -n "$$MLARENA_TEACHER_API_KEY" || { \
		echo "MLARENA_TEACHER_API_KEY is not set (needs a mlk_teacher_... key)."; exit 1; }
	MLARENA_SDK_PATH=$(SDK_PATH) $(PY_COMPS) tools/build_competitions.py attach $(COMP_ARGS)

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
