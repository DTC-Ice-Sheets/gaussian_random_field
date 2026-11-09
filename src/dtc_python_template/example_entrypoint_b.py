"""Entrypoint for package b.

Note that not all packages need a command line entry point.
"""

import logging

import cyclopts

from dtc_python_template.example_sub_package_b.example_module_b import WorseCalculator
from dtc_python_template.monitoring import setup_logging

logger = logging.getLogger(__name__)


app = cyclopts.App()


@app.default
def main(
    integer_a: int,
    integer_b: int,
) -> None:
    """Calculate numbers poorly."""
    setup_logging()
    # Instantiate the calculator and perform calculations
    worse_calculator = WorseCalculator(max_output_array_length=10)
    logger.info(
        "Worse Calculator Output: %s",
        worse_calculator.add_two_numbers(integer_a, integer_b),
    )


if __name__ == "__main__":
    app()
