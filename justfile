set shell := ["zsh", "-cu"]

results-table:
    uv run python scripts/update_readme_results_table.py

model-matrix:
    uv run python scripts/summarize_model_matrix.py
    uv run python scripts/update_readme_model_matrix.py

readme: results-table model-matrix
