"""Brace expansion with the numeric and byte-level core in Mojo.

`bracex` is a recursive-descent expander built out of Python generators. What
is *not* control flow in it is the arithmetic underneath every sequence:

* `{1..10..3}` is a span, a stride and a count: integer arithmetic.
* The zero-pad width of a sequence is a max over two endpoint widths.
* Each value in a sequence becomes zero-padded decimal bytes, which is a
  digit-by-digit loop over a reversed magnitude.
* The cartesian product that stitches sequences back together is a mixed-radix
  decomposition of the row index followed by a byte scatter.

Those four kernels are in `src/kernels.mojo`. The recursive parse, the escape
rules, nesting and the bytes/str conversion stay here in Python, and a
template outside the supported class is handed to the real `bracex` package
rather than guessed at.
"""

from __future__ import annotations

import ctypes
import itertools
import re
import struct
from typing import Any, Iterator

from . import _lib

DEFAULT_LIMIT = 1000
MAX_NEG_INT_64 = -2 ** 63
MAX_INT_64 = abs(MAX_NEG_INT_64 + 1)

# The compiled kernels work on signed 64-bit integers. Upstream's own range
# check allows +2**63, which does not fit, so such a template is forwarded.
_INT64_MIN = -2 ** 63
_INT64_MAX = 2 ** 63 - 1

# Upstream keeps at most 19 digits of a parsed endpoint.
_DIGIT_CAP = 19

# Copied from upstream bracex so the parse cannot drift on its own; the tests
# assert these patterns are identical to the upstream ones.
RE_INT_ITER = re.compile(
    r"(-?((?:0(?=\d))*)\d+)\.{2}(-?((?:0(?=\d))*)\d+)"
    r"(?:\.{2}-?(((?:0(?=\d))*)\d+))?(?=\})"
)
RE_CHR_ITER = re.compile(
    r"([A-Za-z])\.{2}([A-Za-z])(?:\.{2}-?(((?:0(?=\d))*)\d+))?(?=\})"
)

# The alphabet table upstream indexes for `{a..z}` ranges: chr(65)..chr(122)
# with the backslash entry blanked to the empty string. A blanked entry is
# encoded here as codepoint 0, which the kernel turns into a zero-length span.
ALPHA = [0 if x == 0x5C else x for x in range(ord("A"), ord("z") + 1)]
NALPHA = list(reversed(ALPHA))
_ALPHA_BUF = _lib.i32_buffer(len(ALPHA))
_NALPHA_BUF = _lib.i32_buffer(len(NALPHA))
for _i, _v in enumerate(ALPHA):
    _ALPHA_BUF[_i] = _v
for _i, _v in enumerate(NALPHA):
    _NALPHA_BUF[_i] = _v


class ExpansionLimitException(Exception):
    """Brace expansion limit exception."""


def _unpad(group: str, pad_start: int, pad_end: int) -> str:
    """Reproduce upstream's endpoint slicing.

    Upstream keeps the part of the capture before the zero-pad group (the
    sign), drops the pad group, and keeps at most 19 digits of what is left.
    """
    return group[:pad_start] + group[pad_end:pad_end + _DIGIT_CAP]


def _int_group(content: str) -> dict | None:
    """Describe `{a..b[..s]}` the way upstream's regex and span code do."""
    # Upstream matches against the whole template, where the pattern's
    # trailing lookahead sees the closing brace; re-attach it here so the
    # match ends exactly on the group content.
    m = RE_INT_ITER.match(content + "}")
    if m is None or m.end() != len(content):
        return None
    first_s = _unpad(m.group(1), m.start(2) - m.start(1), m.end(2) - m.start(1))
    last_s = _unpad(m.group(3), m.start(4) - m.start(3), m.end(4) - m.start(3))
    if m.group(5) is None:
        inc = 1
    else:
        inc = int(
            _unpad(m.group(5), m.start(6) - m.start(5), m.end(6) - m.start(5))
        )
    first = int(first_s)
    last = int(last_s)
    if inc < 1:
        inc = 1
    if not _INT64_MIN <= first <= _INT64_MAX:
        return None
    if not _INT64_MIN <= last <= _INT64_MAX:
        return None
    count = _lib.lib.bx_int_range_count(first, last, inc)
    if count < 0:
        return None
    spad = m.end(2) - m.start(2)
    epad = m.end(4) - m.start(4)
    return {
        "kind": "int",
        "count": count,
        "first": first,
        "step": inc if first < last else -inc,
        "pad": _lib.lib.bx_pad_width(spad, epad, first, last),
    }


def _char_group(content: str) -> dict | None:
    """Describe `{a..z[..s]}` the way upstream's alphabet table does."""
    m = RE_CHR_ITER.match(content + "}")
    if m is None or m.end() != len(content):
        return None
    start = m.group(1)
    end = m.group(2)
    if m.group(3) is None:
        inc = 1
    else:
        inc = int(
            _unpad(m.group(3), m.start(4) - m.start(3), m.end(4) - m.start(3))
        )
        if inc < 1:
            inc = 1
    inverse = start > end
    table = NALPHA if inverse else ALPHA
    first = table.index(ord(start))
    last = table.index(ord(end))
    count = _lib.lib.bx_char_range_count(first, last, inc)
    if count < 1:
        return None
    return {
        "kind": "chr",
        "count": count,
        "first": first,
        "step": inc if first < last else -inc,
        "inverse": inverse,
    }


def _parse_group(content: str) -> dict | None:
    """One brace group, or None when it is outside the supported class."""
    if not content:
        return None
    group = _int_group(content)
    if group is not None:
        return group
    group = _char_group(content)
    if group is not None:
        return group
    if "," in content:
        parts = content.split(",")
        return {"kind": "list", "count": len(parts), "parts": parts}
    return None


def _scan(template: str) -> tuple[list[str], list[dict]] | None:
    """Split a template into literal segments and one-entry groups.

    Returns None for anything the compiled path does not claim: nesting,
    unbalanced braces, escapes, `$`, and groups that are neither a sequence
    nor a comma list. Upstream expands `{a}` and `{}` to themselves, which is
    recursion rather than arithmetic, so those go to the real package too.
    """
    segments: list[str] = []
    groups: list[dict] = []
    literal: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        c = template[i]
        if c == "\\" or c == "$":
            return None
        if c == "{":
            depth = 0
            k = i
            while k < n:
                if template[k] == "{":
                    depth += 1
                elif template[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                return None
            content = template[i + 1:k]
            if "{" in content or "}" in content:
                return None
            group = _parse_group(content)
            if group is None:
                return None
            segments.append("".join(literal))
            literal = []
            groups.append(group)
            i = k + 1
        else:
            literal.append(c)
            i += 1
    segments.append("".join(literal))
    return segments, groups


def _check_limit(groups: list[dict], limit: int) -> int:
    """Apply upstream's limit to each group, then to the product."""
    total = 1
    for group in groups:
        count = group["count"]
        if limit > 0 and count > limit:
            raise ExpansionLimitException(
                f"Brace expansion has exceeded the limit of {limit:d}"
            )
        total *= count
    if limit > 0 and total > limit:
        raise ExpansionLimitException(
            f"Brace expansion has exceeded the limit of {limit:d}"
        )
    return total



def _dropped_rows(
    segments: list[str], groups: list[dict], rows: int
) -> set[int]:
    """Row indices upstream filters out.

    Upstream carries an empty slot in a comma list as an EMPTY sentinel and
    drops a finished row only when *every* component is that sentinel. Any
    literal text breaks the identity, because upstream keeps literal text as
    a plain `str` and only the empty slot is the sentinel object, so
    `x{a,}b` keeps its `xb` row while `{a,}` does not. A sequence group
    yields plain strings even where a value is empty (upstream's alphabet
    table has a blanked backslash entry), so a template with any range in it
    never drops a row.
    """
    if any(segments):
        return set()
    empties = []
    for group in groups:
        if group["kind"] == "list":
            empties.append([i for i, p in enumerate(group["parts"]) if p == ""])
        else:
            empties.append([])
    if any(not e for e in empties):
        return set()
    n = len(groups)
    stride = [1] * n
    for g in range(n - 2, -1, -1):
        stride[g] = stride[g + 1] * groups[g + 1]["count"]
    drop = set()
    for combo in itertools.product(*empties):
        drop.add(sum(combo[g] * stride[g] for g in range(n)))
    return drop


def _entry_cap(group: dict) -> int:
    """Bytes to reserve for one group in the shared entry buffer."""
    if group["kind"] == "int":
        # A value is at most 20 bytes (19 digits and a sign) or as wide as the
        # pad; one spare byte keeps the estimate generous.
        return group["count"] * (max(group["pad"], 20) + 1)
    if group["kind"] == "chr":
        return group["count"]
    return sum(len(part) for part in group["parts"])


def _widest_entry(group: dict) -> int:
    """Upper bound on the byte length of any one entry of a group."""
    if group["kind"] == "int":
        return max(group["pad"], 20)
    if group["kind"] == "chr":
        return 1
    return max((len(part) for part in group["parts"]), default=0)


def _fill_entries(groups: list[dict]):
    """Materialise every group's entries into one byte buffer.

    Numeric and alphabetic sequences are written by the Mojo kernels, which
    also emit each group's own entry boundary table; comma lists are literal
    text. Returns the entry bytes, the group byte bases, the boundary tables,
    and each group's entry count. The boundary tables are handed to the
    assembly kernel as raw addresses so that no per-entry Python loop is
    needed on the way in.
    """
    caps = [_entry_cap(group) for group in groups]
    entries = _lib.u8_buffer(sum(caps))
    entries_base = _lib.addr(entries)

    grp_off: list[int] = []
    runs = []
    cursor = 0
    for group, cap in zip(groups, caps):
        grp_off.append(cursor)
        count = group["count"]
        if group["kind"] == "list":
            parts = group["parts"]
            blob = "".join(parts).encode("latin-1")
            ctypes.memmove(entries_base + cursor, blob, len(blob))
            run = _lib.i32_buffer(count + 1)
            pos = 0
            run[0] = 0
            for k, part in enumerate(parts):
                pos += len(part)
                run[k + 1] = pos
        else:
            run = _lib.i32_buffer(count + 1)
            if group["kind"] == "int":
                written = _lib.lib.bx_expand_int_range(
                    group["first"], group["step"], count, group["pad"],
                    entries_base + cursor, _lib.addr(run), cap,
                )
            else:
                table = _NALPHA_BUF if group["inverse"] else _ALPHA_BUF
                written = _lib.lib.bx_expand_char_range(
                    group["first"], group["step"], count, _lib.addr(table),
                    entries_base + cursor, _lib.addr(run), cap,
                )
            if written < 0:
                raise RuntimeError("brace expansion entry buffer overflowed")
        runs.append(run)
        cursor += cap
    return entries, grp_off, runs, [g["count"] for g in groups]


def _expand_rows(segments: list[str], groups: list[dict], rows: int) -> list[str]:
    """Cartesian product, as raw latin-1 bytes, via the Mojo row kernel."""
    entries, grp_off, runs, dims = _fill_entries(groups)
    ngrp = len(groups)
    nseg = len(segments)

    lit_blob = "".join(segments).encode("latin-1")
    lit_buf = _lib.u8_buffer(len(lit_blob))
    ctypes.memmove(_lib.addr(lit_buf), lit_blob, len(lit_blob))
    lit_off = _lib.i32_buffer(nseg + 1)
    pos = 0
    for k, segment in enumerate(segments):
        lit_off[k] = pos
        pos += len(segment)
    lit_off[nseg] = pos

    grp_off_buf = _lib.i32_buffer(ngrp + 1)
    for k in range(ngrp):
        grp_off_buf[k] = grp_off[k]
    grp_off_buf[ngrp] = grp_off[-1] if ngrp else 0
    ent_ptr = _lib.i64_buffer(ngrp)
    dim_buf = _lib.i32_buffer(ngrp)
    for k, run in enumerate(runs):
        ent_ptr[k] = _lib.addr(run)
        dim_buf[k] = dims[k]

    # One entry from every group lands in a row, so the widest entries add up
    # rather than compete.
    row_width = pos + sum(_widest_entry(g) for g in groups)
    cap = max(rows * row_width, 1)
    out = _lib.u8_buffer(cap)
    rowlen = _lib.i32_buffer(rows)
    stats = _lib.i32_buffer(2)
    sel = _lib.i32_buffer(2 * ngrp)

    written = _lib.lib.bx_assemble_rows(
        _lib.addr(lit_buf), _lib.addr(lit_off), nseg,
        _lib.addr(entries), _lib.addr(grp_off_buf), _lib.addr(ent_ptr),
        _lib.addr(dim_buf), ngrp,
        0, rows, _lib.addr(sel),
        _lib.addr(out), _lib.addr(rowlen), _lib.addr(stats), cap,
    )
    if written < 0:
        raise RuntimeError("brace expansion output buffer overflowed")

    return _split_rows(out, written, rowlen, stats[0], stats[1])


def _split_rows(out, written: int, rowlen, lo: int, hi: int) -> list[str]:
    """Cut the kernel's flat output into one string per row.

    Decoding the whole block once and slicing the resulting text beats
    decoding each row separately by about a third on a hundred thousand
    rows. When the kernel reports that every row came out the same width --
    a zero-padded sequence, a fixed-width product -- the rows can be cut with
    one strided comprehension and no length table at all, and single-byte
    rows need only `list`.

    The length table is read with `struct` rather than indexed as a ctypes
    array: the same buffer costs about 15 ms to unpack and 380 ms to walk
    element by element at a million rows.
    """
    text = bytes(memoryview(out).cast("B")[:written]).decode("latin-1")
    if lo == hi and written == len(text):
        if lo == 1:
            return list(text)
        return [text[s:s + lo] for s in range(0, written, lo)]
    lengths = struct.unpack("<%di" % len(rowlen), rowlen)
    starts = itertools.accumulate(lengths, initial=0)
    return [text[s:s + length] for s, length in zip(starts, lengths)]


def _split_runs(buf, written: int, offs, count: int) -> list[str]:
    """Cut a kernel's flat output into one string per entry."""
    text = bytes(memoryview(buf).cast("B")[:written]).decode("latin-1")
    bounds = struct.unpack("<%di" % (count + 1), offs)
    return [text[bounds[k]:bounds[k + 1]] for k in range(count)]


def _forward(
    template: str, keep_escapes: bool, limit: int, return_empty: bool
) -> list[str]:
    """Hand a template outside the compiled class to the real package."""
    import bracex

    try:
        return list(bracex.iexpand(template, keep_escapes, limit, return_empty))
    except bracex.ExpansionLimitException as error:
        raise ExpansionLimitException(str(error)) from None


def _expand_str(
    template: str, keep_escapes: bool, limit: int, return_empty: bool
) -> list[str]:
    if not template:
        return [""] if return_empty else []

    scanned = None
    if not keep_escapes:
        try:
            template.encode("latin-1")
        except UnicodeEncodeError:
            scanned = None
        else:
            scanned = _scan(template)

    if scanned is None:
        return _forward(template, keep_escapes, limit, return_empty)

    segments, groups = scanned
    if not groups:
        return [template]
    rows = _check_limit(groups, limit)
    rows_out = _expand_rows(segments, groups, rows)
    drop = _dropped_rows(segments, groups, rows)
    if drop:
        return [e for i, e in enumerate(rows_out) if i not in drop]
    return rows_out


def expand(
    string: Any,
    keep_escapes: bool = False,
    limit: int = DEFAULT_LIMIT,
    return_empty: bool = False,
) -> list[Any]:
    """Expand braces, matching `bracex.expand`.

    A template made only of literal text, comma lists and flat numeric or
    alphabetic sequences is expanded by the Mojo kernels. Everything else is
    forwarded to the real `bracex` package, which owns the recursive parse.
    """
    as_bytes = isinstance(string, bytes)
    template = string.decode("latin-1") if as_bytes else string
    result = _expand_str(template, keep_escapes, limit, return_empty)
    if as_bytes:
        return [entry.encode("latin-1") for entry in result]
    return result


def iexpand(
    string: Any,
    keep_escapes: bool = False,
    limit: int = DEFAULT_LIMIT,
    return_empty: bool = False,
) -> Iterator[Any]:
    """Expand braces and return an iterator, matching `bracex.iexpand`."""
    return iter(expand(string, keep_escapes, limit, return_empty))


def int_range(
    first: int, last: int, inc: int = 1, width: int = 0
) -> list[str]:
    """Values of `{first..last[..inc]}`, the kernel surface on its own.

    `width` is the zero-pad width, applied the way Python's `{:0<pad>d}` does
    it: the sign counts towards the width.
    """
    if inc < 1:
        inc = 1
    count = _lib.lib.bx_int_range_count(first, last, inc)
    if count < 0:
        raise ValueError("range span does not fit in 64 bits")
    cap = count * (max(width, 20) + 1) + 1
    buf = _lib.u8_buffer(cap)
    offs = _lib.i32_buffer(count + 1)
    step = inc if first < last else -inc
    written = _lib.lib.bx_expand_int_range(
        first, step, count, width, _lib.addr(buf), _lib.addr(offs), cap
    )
    if written < 0:
        raise RuntimeError("range output buffer overflowed")
    return _split_runs(buf, written, offs, count)


def char_range(start: str, end: str, inc: int = 1) -> list[str]:
    """Characters of `{start..end[..inc]}` using upstream's alphabet table."""
    if len(start) != 1 or len(end) != 1:
        raise ValueError("alphabetic range endpoints must be single characters")
    if inc < 1:
        inc = 1
    inverse = start > end
    table = NALPHA if inverse else ALPHA
    first = table.index(ord(start))
    last = table.index(ord(end))
    count = _lib.lib.bx_char_range_count(first, last, inc)
    if count < 1:
        return []
    cap = count + 1
    buf = _lib.u8_buffer(cap)
    offs = _lib.i32_buffer(count + 1)
    table_buf = _NALPHA_BUF if inverse else _ALPHA_BUF
    step = inc if first < last else -inc
    written = _lib.lib.bx_expand_char_range(
        first, step, count, _lib.addr(table_buf), _lib.addr(buf),
        _lib.addr(offs), cap,
    )
    if written < 0:
        raise RuntimeError("range output buffer overflowed")
    return _split_runs(buf, written, offs, count)


def range_count(first: int, last: int, inc: int = 1) -> int:
    """How many values upstream's `get_int_range` counts for the range."""
    if inc < 1:
        inc = 1
    return _lib.lib.bx_int_range_count(first, last, inc)


def char_range_count(start: str, end: str, inc: int = 1) -> int:
    """How many characters upstream's `get_char_range` counts."""
    if inc < 1:
        inc = 1
    inverse = start > end
    table = NALPHA if inverse else ALPHA
    return _lib.lib.bx_char_range_count(
        table.index(ord(start)), table.index(ord(end)), inc
    )


def pad_width(spad_len: int, epad_len: int, first: int, last: int) -> int:
    """Zero-pad width upstream's `get_int_range` computes."""
    return _lib.lib.bx_pad_width(spad_len, epad_len, first, last)


def digit_width(value: int) -> int:
    """Length of `str(value)` in decimal, sign included."""
    return _lib.lib.bx_digit_width(value)
