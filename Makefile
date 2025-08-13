# Makefile

# Default target
all: convert-notebooks

# Convert all notebooks to HTML
convert-notebooks:
	@echo "Converting notebooks to HTML..."
	@bash -c "find website/public/modules -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' -exec poetry run jupyter nbconvert --to html {} \;"

.PHONY: all convert-notebooks clean

# Clean generated HTML files
clean:
	@echo "Cleaning up HTML files..."
	@find website/public/modules -name '*.html' -delete
