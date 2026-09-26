"""mojo-bracex: brace expansion with the numeric core in Mojo.

Installable alongside the real `bracex` package, which it is tested against
for parity and which owns the recursive parse for the templates this port
does not claim.
"""

from .core import (
    DEFAULT_LIMIT,
    MAX_INT_64,
    MAX_NEG_INT_64,
    ExpansionLimitException,
    char_range,
    char_range_count,
    digit_width,
    expand,
    iexpand,
    int_range,
    pad_width,
    range_count,
)

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_INT_64",
    "MAX_NEG_INT_64",
    "ExpansionLimitException",
    "char_range",
    "char_range_count",
    "digit_width",
    "expand",
    "iexpand",
    "int_range",
    "pad_width",
    "range_count",
]
__version__ = "0.1.0"
