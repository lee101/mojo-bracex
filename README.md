# mojo-bracex

`mojo-bracex` is the compute-oriented subset of
[bracex](https://github.com/facelessuser/bracex) with the arithmetic and the
byte-level loops behind brace expansion implemented in Mojo and callable from
Python. It keeps the same function shape as `bracex.expand` and the same
output, and the tests compare the two expansion by expansion.

The Python package is `mojo_bracex`, so it installs alongside the real `bracex`
and can be diffed against it directly.

```python
import mojo_bracex as mb

mb.expand("f{a,b}{1..3}.txt")
# ['fa1.txt', 'fa2.txt', 'fa3.txt', 'fb1.txt', 'fb2.txt', 'fb3.txt']
mb.expand("{000001..000010}")
# ['000001', ..., '000010']
```

## What is genuinely compute in bracex

`bracex` is a recursive-descent expander written with Python generators. Most
of it is control flow and belongs to the real package. Four things inside it
are arithmetic, and those are what this port compiles:

| upstream code | what it computes | ported as |
| --- | --- | --- |
| `ExpandBrace.get_int_range` | the span/stride arithmetic behind `{1..10..3}` and the value count | `bx_int_range_count` |
| `ExpandBrace.get_int_range` | `max(spad + len(start), epad + len(end))`, the zero-pad width | `bx_pad_width`, `bx_digit_width` |
| `ExpandBrace.format_values` | each integer as zero-padded decimal bytes | `bx_expand_int_range` |
| `ExpandBrace.get_char_range` | the index walk over the alphabet table | `bx_char_range_count`, `bx_expand_char_range` |
| `ExpandBrace.squash` / `expand_str` | the cartesian product, as a mixed-radix row index plus a byte scatter | `bx_assemble_rows` |

## Covered subset

| area | implemented API |
| --- | --- |
| Whole-template expansion | `expand`, `iexpand` with upstream's `keep_escapes`, `limit`, `return_empty` and `ExpansionLimitException` |
| Templates | literal text, comma lists, flat `{a..b[..s]}` numeric and `{a..z[..s]}` alphabetic sequences, any number of them |
| Sequence semantics | ascending, descending, stride, negative endpoints, implicit zero padding, the alphabet table's blanked backslash entry, empty-slot dropping |
| Kernel surface | `int_range`, `char_range`, `range_count`, `char_range_count`, `pad_width`, `digit_width` |
| Types | `str` and `bytes` input, as upstream: bytes are decoded and re-encoded as latin-1 |

A template outside that class -- nested braces, unbalanced braces, `\` or `$`
escapes, a group that is neither a sequence nor a list (`{a}`, `{}`), or text
that is not latin-1-encodable -- is **forwarded to the real `bracex`**, and
the tests check that the forwarded results are identical. That forwarding is
deliberate: those cases are recursion and lexing, not arithmetic, and
guessing at them would be inventing behaviour.

## Not implemented

- The recursive parse itself: nested groups, `keep_escapes` escape handling,
  `$` look-behind, and the upstream quirk that `{a}` and `{}` expand to
  themselves. All forwarded.
- Endpoints outside the signed 64-bit range. Upstream allows `+2**63` in its
  own check; that value does not fit a signed 64-bit integer, so such a
  template is forwarded rather than silently wrapped.
- A range span wider than 64 bits (`{-2**63..2**63-1}`): the count kernel
  returns a sentinel and the template is forwarded.
- The `__main__` CLI and `__meta__` version plumbing.

## Install and test

```bash
bash build/build.sh          # -> dist/libmojo-bracex.so
PYTHONPATH=python python -m pytest tests -q
```

`bracex` is a runtime dependency only for the forwarded templates; the
covered path imports nothing but the standard library and the shared library.

## Performance

Best-of-N wall clock against the real `bracex`, same process, every case
compared for exact equality before timing. This box is shared, so absolute
times move by tens of percent between runs; the ratios are the stable part.

| case | bracex | mojo-bracex | result |
| --- | ---: | ---: | ---: |
| int range n=100000 | 128.77 ms | 47.72 ms | 2.70x faster |
| padded range n=100000 | 299.10 ms | 99.72 ms | 3.00x faster |
| descending n=100000 | 153.00 ms | 66.93 ms | 2.29x faster |
| product 100x100 | 11.00 ms | 2.46 ms | 4.47x faster |
| three groups 50^3 | 133.89 ms | 64.71 ms | 2.07x faster |
| wide product 100^3 (1e6 rows) | 2456.79 ms | 725.00 ms | 3.39x faster |
| char range n=26 | 0.04 ms | 0.05 ms | 0.87x, slower |

Reproduce with `python bench/bench.py`.

The 26-element alphabetic range is the one loss and it is reported as one: at
that size the work is a single ctypes call plus a 26-element list, so the
fixed cost of crossing the ABI is the whole measurement. The win grows with
the row count because upstream pays generator machinery per row and the port
pays a byte copy.

The floor both implementations share is materialising a Python list of `N`
strings. At a million rows that dominates: the kernel itself is about 100 ms of
the 725 ms, and the rest is building the result list, which cannot be avoided
by either implementation.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bracex.so`.

The Python layer parses the template into literal segments and one-entry
groups, and owns every buffer. Buffers cross the C ABI as 64-bit addresses
and are reconstructed in Mojo as `Pointer[UInt8, AnyOrigin[mut=True]]` and
friends, which keeps the exported symbols non-parametric.

Two details are worth naming because they are easy to get wrong and the tests
pin them:

- Entries inside a group are **not** the same length. `{1..10}` mixes one-byte
  and two-byte values, so the assembly kernel selects an entry by reading a
  boundary table rather than by scaling an index. The count arithmetic and the
  zero-pad width are recomputed from the parsed values, not from the source
  text, so the sign and the stripped leading zeros cannot be lost.
- An empty slot in a comma list is a sentinel that upstream *drops*, but only
  when the whole row is that sentinel. Any literal text in the template breaks
  the identity, so `x{a,}b` keeps `xb` while `{a,}` does not, and a range
  never contributes the sentinel even where its value is the empty string.
  `tests/test_ranges.py` pins both halves of that rule.

## Tolerance

None is needed. Expansion produces decimal text and byte offsets, so every
assertion in the test suite is an exact equality against upstream's output.
There is no floating point in this port, and no FMA question arises.

## License

MIT
