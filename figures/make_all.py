"""Rebuild the paper's illustrations into figures_out/.

    python -m figures.make_all          # all of them
    python -m figures.make_all 2.3 6.2  # only these

Figure 10.1 is a result and comes from experiments/exp_d_compare_methods.py.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figures_out")

FIGURES = [
    ("1.1", "fig_1_1"),
    ("2.1", "fig_2_1"),
    ("2.2", "fig_2_2"),
    ("2.3", "fig_2_3"),
    ("2.4", "fig_2_4"),
    ("3.1", "fig_3_1"),
    ("3.2", "fig_3_2"),
    ("3.3", "fig_3_3"),
    ("5.1", "fig_5_1"),
    ("6.1", "fig_6_1"),
    ("6.2", "fig_6_2"),
    ("7.1", "fig_7_1"),
]


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    for number, module in FIGURES:
        if only and number not in only:
            continue
        print(f"figure {number}")
        importlib.import_module(f"figures.{module}").make()
    if not only:
        print("\nfigure 10.1 is a measurement, not an illustration:")
        print("  python -m experiments.exp_d_compare_methods")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
