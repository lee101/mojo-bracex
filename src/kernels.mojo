"""Compiled inner loops for the numeric part of brace expansion.

`bracex` itself is a recursive-descent expander written with Python
generators. The arithmetic inside it is not: every `{a..z..s}` sequence has a
closed form (the span arithmetic, the zero-pad width, the digit count) and then
a loop that turns integers into zero-padded decimal bytes. The cartesian
product that stitches the sequences back together is a mixed-radix
decomposition followed by a byte scatter. Those five things are what this
compilation unit implements.

Parsing, nesting, escape handling, the expansion limit and the bytes/str
conversion stay in Python, because they are control flow rather than compute.

Every exported symbol takes buffer addresses as plain `Int` values and
rebuilds the pointer inside the body, because `@export` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.
"""

comptime I32P = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64P = Pointer[Int64, AnyOrigin[mut=True]]
comptime U8P = Pointer[UInt8, AnyOrigin[mut=True]]


def i32p(addr: Int) -> I32P:
    return I32P(unsafe_from_address=addr)


def i64p(addr: Int) -> I64P:
    return I64P(unsafe_from_address=addr)


def u8p(addr: Int) -> U8P:
    return U8P(unsafe_from_address=addr)


def magnitude(v: Int) -> UInt64:
    """Unsigned magnitude of a signed 64-bit value.

    Negating `Int64.min` overflows, so the sign is folded in before the
    negation. The Python side rejects anything outside the signed 64-bit
    range, so this never sees an unrepresentable input.
    """
    if v < 0:
        return UInt64(-(v + 1)) + UInt64(1)
    return UInt64(v)


def digit_count(mag: UInt64) -> Int:
    """Decimal digits in `mag`, counting zero as one digit."""
    var n = 1
    var q = mag
    while q >= UInt64(10):
        q = q // UInt64(10)
        n += 1
    return n


def pow10(e: Int) -> UInt64:
    var r = UInt64(1)
    var i = 0
    while i < e:
        r = r * UInt64(10)
        i += 1
    return r


def signed_digits(v: Int) -> Int:
    """Length of the decimal spelling of `v`, sign included."""
    var n = digit_count(magnitude(v))
    if v < 0:
        n += 1
    return n


@export("bx_digit_width")
def bx_digit_width(v: Int) abi("C") -> Int:
    """Length of `str(v)` in decimal, sign included, zero counted as one."""
    return signed_digits(v)


@export("bx_pad_width")
def bx_pad_width(spad_len: Int, epad_len: Int, first: Int, last: Int) abi("C") -> Int:
    """Zero-pad width for an integer range, matching upstream bracex.

    Upstream takes `max(spad_len + len(start), epad_len + len(end))` when
    either endpoint carried an explicit zero pad, and 0 otherwise, where
    `spad_len` is the number of leading zeros that were stripped off the
    first endpoint and `len(start)` counts the sign. Both spellings are
    derived from the value here, so the sign and the leading-zero group of
    the source literal cannot be forgotten.
    """
    if spad_len == 0 and epad_len == 0:
        return 0
    var a = spad_len + signed_digits(first)
    var b = epad_len + signed_digits(last)
    if a >= b:
        return a
    return b


@export("bx_int_range_count")
def bx_int_range_count(first: Int, last: Int, inc: Int) abi("C") -> Int:
    """Number of values bracex would emit for `{first..last[..inc]}`.

    `inc` is already `max(1, parsed)` as upstream requires. Returns -1 when
    the span does not fit in a signed 64-bit integer, which the Python side
    treats as "too large to expand here" and forwards to upstream.
    """
    var span: Int
    if first < last:
        span = last - first
        if span < 0:
            return -1
        span += 1
    else:
        span = first - last
        if span < 0:
            return -1
        span += 1
    var ainc = inc
    if ainc < 0:
        ainc = -ainc
    if ainc == 0:
        return 1
    if ainc > span:
        return 1
    var count = span // ainc
    if span % ainc != 0:
        count += 1
    return count


@export("bx_char_range_count")
def bx_char_range_count(first: Int, last: Int, inc: Int) abi("C") -> Int:
    """Number of characters bracex would emit for an alphabetic range.

    The endpoints are indices into the alphabet table, so the count is the
    same span arithmetic as the integer case.
    """
    var span: Int
    if first < last:
        span = last - first + 1
    else:
        span = first - last + 1
    if inc == 0:
        return 1
    if inc > span:
        return 1
    var count = span // inc
    if span % inc != 0:
        count += 1
    return count


@export("bx_expand_int_range")
def bx_expand_int_range(
    first: Int, step: Int, count: Int, pad: Int,
    out_addr: Int, off_addr: Int, cap: Int
) abi("C") -> Int:
    """Write `count` values `first, first+step, ...` as padded decimal bytes.

    Returns the number of bytes written, or -1 if they do not fit in `cap`.
    `off_addr` receives `count + 1` Int32 byte offsets, entry boundaries, so
    the caller can splice the values into a larger row without a second pass
    over the bytes.
    """
    var out = u8p(out_addr)
    var offs = i32p(off_addr)
    var pos = 0
    offs[unsafe_offset=0] = Int32(0)
    var v = first
    var i = 0
    while i < count:
        var negative = v < 0
        var mag = magnitude(v)
        var dl = digit_count(mag)
        var total = dl
        if negative:
            total += 1
        var zeros = pad - total
        if zeros < 0:
            zeros = 0
        if pos + total + zeros > cap:
            return -1
        if negative:
            out[unsafe_offset=pos] = UInt8(45)
            pos += 1
        var z = 0
        while z < zeros:
            out[unsafe_offset=pos] = UInt8(48)
            pos += 1
            z += 1
        var div = pow10(dl - 1)
        var d = 0
        while d < dl:
            out[unsafe_offset=pos] = UInt8(48 + (mag // div) % UInt64(10))
            pos += 1
            div = div // UInt64(10)
            d += 1
        offs[unsafe_offset=i + 1] = Int32(pos)
        v = v + step
        i += 1
    return pos


@export("bx_expand_char_range")
def bx_expand_char_range(
    start_index: Int, step: Int, count: Int, alpha_addr: Int,
    out_addr: Int, off_addr: Int, cap: Int
) abi("C") -> Int:
    """Walk `count` alphabet entries and write their single bytes.

    `alpha_addr` is the 58-entry table upstream builds from chr(65)..chr(122)
    with the backslash entry blanked. A blanked entry is encoded here as
    codepoint 0 and contributes a zero-length span, which is exactly the empty
    string upstream yields for that index.
    """
    var alpha = i32p(alpha_addr)
    var out = u8p(out_addr)
    var offs = i32p(off_addr)
    var pos = 0
    offs[unsafe_offset=0] = Int32(0)
    var idx = start_index
    var i = 0
    while i < count:
        if pos >= cap:
            return -1
        var cp = alpha[unsafe_offset=idx]
        if cp > 0:
            out[unsafe_offset=pos] = UInt8(cp)
            pos += 1
        offs[unsafe_offset=i + 1] = Int32(pos)
        idx = idx + step
        i += 1
    return pos


@export("bx_assemble_rows")
def bx_assemble_rows(
    lit_addr: Int, lit_off_addr: Int, nseg: Int,
    grp_addr: Int, grp_off_addr: Int, grp_ent_ptr: Int,
    grp_dim_addr: Int, ngrp: Int,
    row0: Int, row1: Int, sel_addr: Int,
    out_addr: Int, rowlen_addr: Int, stats_addr: Int, cap: Int
) abi("C") -> Int:
    """Cartesian product of `ngrp` groups, as raw latin-1 bytes.

    Row `r` is `seg[0] + entry(g0,r) + seg[1] + entry(g1,r) + ...`. The row
    index is decomposed as a mixed-radix number with the last group varying
    fastest, which is the order `itertools.product` produces upstream.

    Entries within a group are not all the same length: `{1..10}` mixes
    one-byte and two-byte values. So the selected entry is read from a
    boundary table rather than scaled. `grp_ent_ptr` is an Int64 array whose
    `g`th slot is the address of group `g`'s own Int32 boundary table, as
    produced by `bx_expand_int_range` or `bx_expand_char_range`, and
    `grp_off[g]` is where that group's bytes start in `grp`.

    `sel_addr` is scratch of at least `2 * ngrp` Int32 slots holding the
    selected entry's [start, end) byte range per group.

    `stats_addr` receives two Int32 slots, the narrowest and widest row, so
    the caller can take a fixed-width fast path when every row is the same
    size.

    Returns the total bytes written, or -1 if the rows do not fit in `cap`.
    """
    var lit = u8p(lit_addr)
    var litoff = i32p(lit_off_addr)
    var grp = u8p(grp_addr)
    var groff = i32p(grp_off_addr)
    var entptr = i64p(grp_ent_ptr)
    var gdim = i32p(grp_dim_addr)
    var sel = i32p(sel_addr)
    var out = u8p(out_addr)
    var rowlen = i32p(rowlen_addr)
    var stats = i32p(stats_addr)
    var lo_len = 0x7FFFFFFF
    var hi_len = 0
    var pos = 0
    var r = row0
    while r < row1:
        var row_start = pos
        # Decompose from the last group backwards, so group 0 is the most
        # significant digit and the last group varies fastest, which is the
        # order `itertools.product` produces upstream.
        var k = r
        var g = ngrp - 1
        while g >= 0:
            var d = Int(gdim[unsafe_offset=g])
            var e = 0
            if d > 1:
                e = k % d
                k = k // d
            var entoff = i32p(Int(entptr[unsafe_offset=g]))
            var gstart = Int(groff[unsafe_offset=g])
            sel[unsafe_offset=2 * g] = Int32(
                gstart + Int(entoff[unsafe_offset=e])
            )
            sel[unsafe_offset=2 * g + 1] = Int32(
                gstart + Int(entoff[unsafe_offset=e + 1])
            )
            g -= 1
        var s = 0
        while s < nseg:
            var lo = Int(litoff[unsafe_offset=s])
            var hi = Int(litoff[unsafe_offset=s + 1])
            var c = lo
            while c < hi:
                if pos >= cap:
                    return -1
                out[unsafe_offset=pos] = lit[unsafe_offset=c]
                pos += 1
                c += 1
            if s < ngrp:
                var gs = Int(sel[unsafe_offset=2 * s])
                var ge = Int(sel[unsafe_offset=2 * s + 1])
                var c2 = gs
                while c2 < ge:
                    if pos >= cap:
                        return -1
                    out[unsafe_offset=pos] = grp[unsafe_offset=c2]
                    pos += 1
                    c2 += 1
            s += 1
        var width = pos - row_start
        rowlen[unsafe_offset=r - row0] = Int32(width)
        if width < lo_len:
            lo_len = width
        if width > hi_len:
            hi_len = width
        r += 1
    stats[unsafe_offset=0] = Int32(lo_len)
    stats[unsafe_offset=1] = Int32(hi_len)
    return pos
