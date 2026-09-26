"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay `c_int64` for addresses;
`c_int` truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bracex.so"

_I64 = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.bx_digit_width.restype = ctypes.c_int64
    lib.bx_digit_width.argtypes = [_I64]

    lib.bx_pad_width.restype = ctypes.c_int64
    lib.bx_pad_width.argtypes = [_I64, _I64, _I64, _I64]

    lib.bx_int_range_count.restype = ctypes.c_int64
    lib.bx_int_range_count.argtypes = [_I64, _I64, _I64]

    lib.bx_char_range_count.restype = ctypes.c_int64
    lib.bx_char_range_count.argtypes = [_I64, _I64, _I64]

    lib.bx_expand_int_range.restype = ctypes.c_int64
    lib.bx_expand_int_range.argtypes = [
        _I64, _I64, _I64, _I64, _I64, _I64, _I64,
    ]

    lib.bx_expand_char_range.restype = ctypes.c_int64
    lib.bx_expand_char_range.argtypes = [
        _I64, _I64, _I64, _I64, _I64, _I64, _I64,
    ]

    lib.bx_assemble_rows.restype = ctypes.c_int64
    lib.bx_assemble_rows.argtypes = [_I64] * 15
    return lib

lib = _load()


def u8_buffer(nbytes: int):
    """A writable byte buffer that keeps itself alive while it is addressed."""
    return (ctypes.c_uint8 * max(nbytes, 1))()


def i32_buffer(n: int):
    """A writable int32 buffer that keeps itself alive while it is addressed."""
    return (ctypes.c_int32 * max(n, 1))()


def i64_buffer(n: int):
    """A writable int64 buffer that keeps itself alive while it is addressed."""
    return (ctypes.c_int64 * max(n, 1))()


def i32_buffer(n: int):
    """A writable int32 buffer that keeps itself alive while it is addressed."""
    return (ctypes.c_int32 * max(n, 1))()


def addr(buf) -> int:
    return ctypes.addressof(buf)
