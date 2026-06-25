"""Write the canonical README/Page Plotly figure as PNG and SVG."""
from __future__ import annotations

import readme_plot


def main() -> None:
    readme_plot.write_main_plot_assets()
    print(readme_plot.MAIN_PNG)
    print(readme_plot.MAIN_SVG)


if __name__ == "__main__":
    main()
