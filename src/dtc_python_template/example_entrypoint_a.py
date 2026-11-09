"""Entrypoint for subpackage a.

Note that not all packages need a command line entry point.
"""

import logging

import cyclopts

from dtc_python_template.example_sub_package_a.example_module_a import BadCalculator
from dtc_python_template.monitoring import setup_logging

logger = logging.getLogger(__name__)


app = cyclopts.App()


@app.default
def main(
    integer_a: int,
    integer_b: int,
) -> None:
    """Calculate numbers poorly, and log the result.

    Parameters
    ----------
    integer_a : int
        The first integer.
    integer_b : int
        The second integer.
    """
    setup_logging()
    # Instantiate the calculator and perform calculations
    bad_calculator = BadCalculator(output_array_length=5)
    logger.info(
        "Bad Calculator Output: %s",
        bad_calculator.add_two_numbers(integer_a, integer_b),
    )


if __name__ == "__main__":
    app()
