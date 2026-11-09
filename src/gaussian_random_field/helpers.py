"""Utility functions for use across use_case_1 modules."""

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

logger = logging.getLogger(__name__)


def path_or_web_url_validator(type_: object, path_or_url: Path | str) -> str:  # ruff:ignore[unused-function-argument] - needed type_ to match cyclopts validator signature
    """Validate that the provided path is either a valid local path or a web URL.

    NOTE: Path("") resolves to the current directory (".") - which is considered a valid path.

    Parameters
    ----------
    type_ : object
        The type of the argument being validated - unused, purely required for cyclopts validator compatibility.
    path_or_url : Path | str
        The candidate path or URL to validate.

    Returns
    -------
    str
        The validated path or URL as a string.

    Raises
    ------
    ValueError
        If the provided path is neither a valid local path nor a web URL.
    """
    if not Path(path_or_url).exists() and not str(path_or_url).startswith(("http://", "https://")):
        raise ValueError(f"The provided path '{path_or_url}' is neither a valid local path nor a web URL.")
    return str(path_or_url)


# alias for use in cyclopts Parameter definitions
type ZarrType = Annotated[str, Parameter(validator=path_or_web_url_validator)]
