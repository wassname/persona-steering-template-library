set shell := ["zsh", "-cu"]

results-table:
    just readme

model-matrix:
    just readme

readme:
    uv run python scripts/summarize_model_matrix.py
    PSTL_DOC_TARGET=gfm QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render README.qmd --to gfm

pages:
    uv run python scripts/summarize_model_matrix.py
    PSTL_DOC_TARGET=html QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render README.qmd --to html --output-dir docs/_site --output index.html
