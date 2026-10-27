"""A block-granular scheduler rounds the prefix boundary down to a block multiple, so '4096' means 4096..4096+block-1
exact tokens."""
from kva.kva import rounded_tail


def test_rounded_tail_matches_block_rounding():
    assert rounded_tail(18984, 4096, 1568) == 18984 - (18984 - 4096) // 1568 * 1568
    assert 4096 <= rounded_tail(18984, 4096, 1568) < 4096 + 1568


def test_rounded_tail_short_prompt_keeps_whole_prompt_exact():
    assert rounded_tail(3000, 4096, 1568) == 3000


def test_rounded_tail_exact_multiple_gives_base():
    assert rounded_tail(4096 + 3 * 1568, 4096, 1568) == 4096
