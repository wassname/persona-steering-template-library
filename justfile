set shell := ["zsh", "-cu"]

results-table:
    just readme

model-matrix:
    just readme

readme:
    uv run python scripts/summarize_model_matrix.py
    QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render README.qmd --to gfm

pages:
    uv run python scripts/summarize_model_matrix.py
    QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render docs/index.qmd --to html --output-dir _site
    mkdir -p docs/_site/out/model_matrix
    cp out/model_matrix/refusal_probe_seed24_n1_model_matrix.svg docs/_site/out/model_matrix/
