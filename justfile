set shell := ["zsh", "-cu"]

results-table:
    uv run python scripts/update_readme_results_table.py

model-matrix:
    uv run python scripts/summarize_model_matrix.py
    uv run python scripts/update_readme_model_matrix.py

readme:
    uv run python scripts/summarize_model_matrix.py
    QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render README.qmd --to gfm
