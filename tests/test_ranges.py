"""Parity of the compiled range kernels against the real `bracex`.

Every assertion here is an equality against upstream's own output. That is
exact, not tolerant: the port produces decimal text and byte offsets, so there
is no floating point anywhere in the comparison and an FMA cannot make a
correct kernel look broken.
"""

import pytest

import bracex

import mojo_bracex as mb
from mojo_bracex import core as mbc


INT_TEMPLATES = [
    "{1..3}",
    "{0..0}",
    "{1..1}",
    "{3..1}",
    "{5..1}",
    "{1..10}",
    "{1..10..3}",
    "{1..10..2}",
    "{10..1..3}",
    "{1..1000}",
    "{0..255}",
    "{-3..2}",
    "{-3..2..2}",
    "{-1..-5}",
    "{01..10}",
    "{001..1000..7}",
    "{-05..-01}",
    "{1..10..-2}",
    "{10..1..-2}",
    "{1..10..100}",
    "{5..1..2}",
    "{100..0..25}",
    "{9223372036854775800..9223372036854775805}",
    "{-9223372036854775805..-9223372036854775800}",
]

CHAR_TEMPLATES = [
    "{a..e}",
    "{a..e..2}",
    "{a..z}",
    "{a..z..5}",
    "{z..a}",
    "{y..a..3}",
    "{A..z}",
    "{Z..a}",
    "{A..a}",
    "{a..a}",
    "{b..z..7}",
    "{0..0}",
]

MIXED_TEMPLATES = [
    "f{a,b}{1..3}.txt",
    "x{1..3}y{b,c}",
    "a{1,2}{3,4}",
    "{1..3}{4,5}",
    "x{1..3}{a,b}{9,8}y",
    "{,a}{1..2}",
    "{,}{1..2}",
    "x{1..2}{,}",
    "{Z..a}{,q}",
    "{,ab}",
    "prefix_{1..3}_{a,b}_{0..2..1}_suffix",
]

# Templates the compiled path declines: recursion, escapes, or a group that is
# neither a sequence nor a list. These must still match, via the forward.
FORWARDED_TEMPLATES = [
    "{a}",
    "{}",
    "{a{b,c}}",
    "{a{b,c},d}",
    "\\{1..2}",
    "${1..2}",
    "$x{1..2}",
    "{a",
    "{1..3",
    "a}b",
    "{1..3}}",
    "{a..b..c}",
    "{1..3,5}",
    "{1..3,5,7}",
]


def _run(fn, template, **kwargs):
    try:
        return ("ok", fn(template, **kwargs))
    except Exception as error:  # noqa: BLE001 - the type is part of the contract
        return ("err", type(error).__name__, str(error))


@pytest.mark.parametrize("template", INT_TEMPLATES + CHAR_TEMPLATES)
def test_range_parity(template):
    """Every sequence form expands to upstream's exact list of strings."""
    assert mb.expand(template) == bracex.expand(template)


@pytest.mark.parametrize("template", MIXED_TEMPLATES)
def test_cartesian_product_parity(template):
    """Groups mix into rows in upstream's `itertools.product` order."""
    assert mb.expand(template) == bracex.expand(template)


@pytest.mark.parametrize("template", FORWARDED_TEMPLATES)
def test_forwarded_templates_still_match(template):
    """Templates outside the compiled class are handed to upstream."""
    assert mb.expand(template) == bracex.expand(template)


def test_empty_slot_is_dropped_only_when_it_is_the_whole_row():
    """Upstream drops a row only when every component is its EMPTY sentinel.

    A plain literal breaks the identity, so `x{a,}b` keeps `xb` while
    `{a,}` loses its empty row, and a range never contributes the sentinel.
    """
    assert mb.expand("{a,}") == bracex.expand("{a,}") == ["a"]
    assert mb.expand("x{a,}b") == bracex.expand("x{a,}b") == ["xab", "xb"]
    assert mb.expand("a{,}b") == bracex.expand("a{,}b") == ["ab", "ab"]
    assert mb.expand("{,a}{1..2}") == bracex.expand("{,a}{1..2}") == [
        "1", "2", "a1", "a2",
    ]
    # The blanked backslash in upstream's alphabet table yields a real empty
    # string, not the sentinel, so the row survives.
    assert "" in mb.expand("{Z..a}")


def test_last_group_varies_fastest():
    """Row order is the digit order, not the group order."""
    assert mb.expand("a{1,2}{3,4}") == ["a13", "a14", "a23", "a24"]


def test_variable_length_entries_in_one_group():
    """`{1..10}` mixes one-byte and two-byte entries in a single group.

    A kernel that assumed a uniform entry width would scramble this.
    """
    assert mb.expand("x{1..10}") == bracex.expand("x{1..10}")
    assert mb.expand("{1..10}y") == bracex.expand("{1..10}y")
    assert mb.expand("{1..10}{5..6}") == bracex.expand("{1..10}{5..6}")


def test_zero_pad_width_counts_the_sign():
    """Python's `{:0<pad>d}` puts the zeros after the sign."""
    assert mb.expand("{-05..-01}") == ["-05", "-04", "-03", "-02", "-01"]
    assert mb.expand("{01..10}") == ["01", "02", "03", "04", "05",
                                     "06", "07", "08", "09", "10"]


def test_int_range_kernel_directly():
    assert mb.int_range(1, 10, 3) == ["1", "4", "7", "10"]
    assert mb.int_range(5, 1, 2) == ["5", "3", "1"]
    assert mb.int_range(-3, 3, 3) == ["-3", "0", "3"]
    assert mb.int_range(0, 0) == ["0"]


def test_int_range_kernel_pad_width():
    assert mb.int_range(1, 12, 5, width=4) == ["0001", "0006", "0011"]
    assert mb.int_range(-3, 3, 6, width=4) == ["-003", "0003"]


def test_char_range_kernel_directly():
    assert mb.char_range("a", "e", 2) == ["a", "c", "e"]
    assert mb.char_range("y", "a", 3) == ["y", "v", "u", "t", "s", "r", "q",
                                          "p", "o", "n", "m", "l", "k", "j",
                                          "i", "h", "g", "f", "e", "d", "c",
                                          "b", "a"]
    assert mb.char_range("A", "a") == bracex.expand("{A..a}")


def test_range_count_matches_python_len():
    """The count kernel must agree with how many values actually come out."""
    for first, last, inc in [
        (1, 10, 1), (1, 10, 3), (10, 1, 3), (0, 0, 1), (-5, 5, 2),
        (1, 1, 1), (1, 1000, 7), (-1, -100, 13), (1, 2, 1000),
    ]:
        got = mb.range_count(first, last, inc)
        assert got == len(mb.int_range(first, last, inc)), (first, last, inc)
        assert got == len(bracex.expand("{%d..%d..%d}" % (first, last, inc)))


def test_char_range_count_matches_python_len():
    for start, end, inc in [("a", "z", 1), ("a", "z", 4), ("z", "a", 5),
                            ("A", "z", 3), ("b", "b", 1), ("a", "c", 99)]:
        got = mb.char_range_count(start, end, inc)
        assert got == len(mb.char_range(start, end, inc))
        assert got == len(bracex.expand("{%s..%s..%d}" % (start, end, inc)))


def test_char_range_kernel_directly():
    assert mb.char_range("a", "e", 2) == ["a", "c", "e"]
    assert mb.char_range("y", "a", 3) == bracex.expand("{y..a..3}")
    assert mb.char_range("A", "a") == bracex.expand("{A..a}")


def test_pad_width_kernel_matches_upstream_rule():
    """`max(spad + len(start), epad + len(end))`, 0 when neither endpoint pads."""
    for spad, epad, first, last in [
        (0, 0, 1, 10), (1, 0, 1, 10), (2, 0, 1, 1000), (0, 2, 1, 1000),
        (1, 1, -5, -1), (3, 0, -5, 2), (0, 0, -1, -100),
    ]:
        if spad == 0 and epad == 0:
            expect = 0
        else:
            expect = max(spad + len(str(first)), epad + len(str(last)))
        assert mb.pad_width(spad, epad, first, last) == expect


def test_digit_width_includes_the_sign():
    for value in (0, 1, -1, 9, 10, -10, 999999, -1000000, 2**62):
        assert mb.digit_width(value) == len(str(value))


def test_range_count_rejects_a_span_that_overflows_64_bits():
    assert mb.range_count(-(2**63), 2**63 - 1, 1) == -1
    with pytest.raises(ValueError):
        mb.int_range(-(2**63), 2**63 - 1, 1)


def test_expansion_limit_message_matches_upstream():
    """The limit is reported the same way, from the same rule."""
    for limit, template in [(1000, "{1..100000}"), (3, "{1..5}"),
                            (2, "{a,b}{c,d}"), (4, "{a,b}{c,d}"),
                            (1, "{1..2}")]:
        mine = _run(mb.expand, template, limit=limit)
        theirs = _run(bracex.expand, template, limit=limit)
        assert mine[0] == theirs[0]
        if mine[0] == "err":
            assert mine[1:] == theirs[1:]


def test_limit_disabled_agrees_on_a_big_template():
    big = "x{1..300}y"
    assert mb.expand(big, limit=0) == bracex.expand(big, limit=0)


def test_bytes_input_round_trips():
    assert mb.expand(b"f{a,b}{1..3}.txt") == bracex.expand(b"f{a,b}{1..3}.txt")
    assert mb.expand(b"{a,,b}") == bracex.expand(b"{a,,b}")


def test_iexpand_matches_expand():
    assert list(mb.iexpand("{1..3}")) == mb.expand("{1..3}")
    assert list(mb.iexpand(b"{1..3}")) == list(bracex.iexpand(b"{1..3}"))


def test_empty_template_honours_return_empty():
    assert mb.expand("") == bracex.expand("") == []
    assert mb.expand("", return_empty=True) == [""] == bracex.expand(
        "", return_empty=True
    )


def test_keep_escapes_is_forwarded():
    assert mb.expand("\\{1..2}", keep_escapes=True) == bracex.expand(
        "\\{1..2}", keep_escapes=True
    )
    assert mb.expand("\\{1..2}") == bracex.expand("\\{1..2}")


def test_parse_patterns_are_upstreams():
    """The port copies the two upstream regexes; they must not drift."""
    assert mbc.RE_INT_ITER.pattern == bracex.RE_INT_ITER.pattern
    assert mbc.RE_CHR_ITER.pattern == bracex.RE_CHR_ITER.pattern
    assert mbc.ALPHA == [0 if x == 0x5C else x
                         for x in range(ord("A"), ord("z") + 1)]


def test_a_larger_expansion_is_exactly_right():
    """One template with a wide, mixed shape, checked value by value."""
    template = "run{01..12}_{a,c}_{0..3..1}"
    got = mb.expand(template, limit=1000)
    assert got == bracex.expand(template, limit=1000)
    assert len(got) == 12 * 2 * 4
    assert got[0] == "run01_a_0"
    assert got[-1] == "run12_c_3"
    assert len(set(got)) == len(got)


def test_fuzz_against_upstream():
    """A deterministic sweep of mixed templates through both implementations."""
    groups = ["{1..3}", "{2..5}", "{a..c}", "{1,2}", "{,b}", "{a,}",
              "{0..9..2}", "{c..a}", "{-2..2}", "{01..04}", "{5..1}",
              "{a..d..2}"]
    literals = ["", "x", "file", "-", "0", "zz"]
    checked = 0
    for i in range(400):
        parts = []
        for _ in range(1 + (i % 3)):
            parts.append(literals[i % len(literals)])
            parts.append(groups[(i // 2) % len(groups)])
        parts.append(literals[i % len(literals)])
        template = "".join(parts)
        if mbc._scan(template) is None:
            continue
        assert mb.expand(template, limit=1000) == bracex.expand(
            template, limit=1000
        ), template
        checked += 1
    assert checked > 300
