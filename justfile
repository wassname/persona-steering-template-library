set shell := ["zsh", "-cu"]

results-table:
    uv run python scripts/update_readme_results_table.py
