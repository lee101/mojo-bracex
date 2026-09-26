"""Correctness-gated benchmark for mojo-bracex.

Every case expands through the real `bracex` as well and compares the result
before timing anything, so a regression in the Mojo kernels shows up as a
mismatch rather than as a suspiciously good number. Expansion is exact string
work, so the gate is equality, not a tolerance.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import bracex  # noqa: E402

import mojo_bracex as mb  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_int_range(count: int = 100_000):
    """A single wide sequence, the kernel's own hot path."""
    template = "{1..%d}" % count
    mine = mb.expand(template, limit=0)
    theirs = bracex.expand(template, limit=0)
    assert mine == theirs, "int range mismatch"
    assert len(mine) == count
    return (
        "int range n=%d" % count,
        _time(lambda: bracex.expand(template, limit=0), 3),
        _time(lambda: mb.expand(template, limit=0), 3),
    )


def bench_padded_range(count: int = 100_000):
    """A zero-padded sequence: digit emission plus padding per value."""
    template = "{000001..%06d}" % count
    mine = mb.expand(template, limit=0)
    assert mine == bracex.expand(template, limit=0), "padded range mismatch"
    return (
        "padded range n=%d" % count,
        _time(lambda: bracex.expand(template, limit=0), 3),
        _time(lambda: mb.expand(template, limit=0), 3),
    )

def bench_product(a: int = 100, b: int = 100):
    """A two-group product: the mixed-radix decomposition and byte scatter."""
    template = "src/file{0..%d}_{0..%d}.txt" % (a - 1, b - 1)
    mine = mb.expand(template, limit=0)
    assert mine == bracex.expand(template, limit=0), "product mismatch"
    assert len(mine) == a * b
    return (
        "product %dx%d" % (a, b),
        _time(lambda: bracex.expand(template, limit=0), 3),
        _time(lambda: mb.expand(template, limit=0), 3),
    )


def bench_descending(count: int = 100_000):
    """A descending sequence: the same loop with a negative stride."""
    template = "{%d..1}" % count
    assert mb.expand(template, limit=0) == bracex.expand(template, limit=0)
    return (
        "descending n=%d" % count,
        _time(lambda: bracex.expand(template, limit=0), 3),
        _time(lambda: mb.expand(template, limit=0), 3),
    )


def bench_char_range(count: int = 26):
    """An alphabetic sequence: the alphabet table walk, one byte per value."""
    template = "{%s..%s}" % ("a", chr(ord("a") + count - 1))
    assert mb.expand(template, limit=0) == bracex.expand(template, limit=0)
    return (
        "char range n=%d" % count,
        _time(lambda: bracex.expand(template, limit=0)),
        _time(lambda: mb.expand(template, limit=0)),
    )


def bench_three_groups(n: int = 50):
    """Three groups, so the radix has more than two levels."""
    template = "a{0..%d}b{0..%d}c{0..%d}" % (n - 1, n - 1, n - 1)
    assert mb.expand(template, limit=0) == bracex.expand(template, limit=0)
    return (
        "three groups n=%d^3" % n,
        _time(lambda: bracex.expand(template, limit=0), 3),
        _time(lambda: mb.expand(template, limit=0), 3),
    )


def bench_wide_product(n: int = 100):
    """A million rows of mixed width, where building the result list is the
    floor both implementations pay."""
    template = "a{0..%d}b{0..%d}c{0..%d}" % (n - 1, n - 1, n - 1)
    assert mb.expand(template, limit=0) == bracex.expand(template, limit=0)
    assert len(mb.expand(template, limit=0)) == n ** 3
    return (
        "wide product %d^3" % n,
        _time(lambda: bracex.expand(template, limit=0), 2),
        _time(lambda: mb.expand(template, limit=0), 2),
    )


def main():
    print(f"{'case':<24}{'bracex':>12}{'mojo-bracex':>14}{'ratio':>9}")
    print("-" * 60)
    for fn in (
        bench_int_range,
        bench_padded_range,
        bench_descending,
        bench_char_range,
        bench_product,
        bench_three_groups,
        bench_wide_product,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<24}{ref*1e3:>10.2f}ms{got*1e3:>12.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
