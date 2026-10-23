import hashlib
import json
import re
import subprocess
import sys
import unicodedata

import numpy as np
import pytest

from lookalike_texts import (
    MINUTES,
    REGULATION_V1,
    REGULATION_V2,
    SPEC_RECRUITING,
    SPEC_WAREHOUSE,
)
from minirag_mcp.lookalike import (
    MAX_LOOK_ALIKES,
    NUM_PERM,
    THRESHOLD,
    SignatureIndex,
    overlap,
    signature,
)

_WORD = re.compile(r"\w+")


def exact_jaccard(a: str, b: str) -> float:
    def shingles(text):
        words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
        return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}

    x, y = shingles(a), shingles(b)
    return len(x & y) / len(x | y)


def test_two_versions_of_one_regulation_overlap_above_the_threshold():
    exact = exact_jaccard(REGULATION_V1, REGULATION_V2)
    estimate = overlap(signature(REGULATION_V1), signature(REGULATION_V2))
    assert exact > 0.6
    assert estimate >= THRESHOLD
    assert abs(estimate - exact) < 0.1


def test_two_specifications_from_one_template_stay_below_the_threshold():
    """Same headings and boilerplate, different substance: a sibling, not a version."""
    exact = exact_jaccard(SPEC_WAREHOUSE, SPEC_RECRUITING)
    estimate = overlap(signature(SPEC_WAREHOUSE), signature(SPEC_RECRUITING))
    assert 0.1 < exact < 0.4  # they do share text, just not most of it
    assert estimate < THRESHOLD
    assert abs(estimate - exact) < 0.1


def test_unrelated_documents_share_nothing():
    assert overlap(signature(REGULATION_V1), signature(MINUTES)) < 0.05
    assert overlap(signature(SPEC_WAREHOUSE), signature(REGULATION_V1)) < 0.05


def test_a_document_is_identical_to_itself():
    assert overlap(signature(REGULATION_V1), signature(REGULATION_V1)) == 1.0


def test_case_and_unicode_composition_do_not_change_a_signature():
    """APFS hands names and extracted text back decomposed; a model types them composed."""
    nfd = unicodedata.normalize("NFD", REGULATION_V1.upper())
    assert nfd != REGULATION_V1
    a, b = signature(REGULATION_V1), signature(nfd)
    assert np.array_equal(a.values, b.values) and a.shingles == b.shingles


def test_markup_and_spacing_between_words_do_not_count():
    plain = "Отдел закупок проверяет комплектность заявки в течение пяти рабочих дней"
    marked = (
        "**Отдел закупок** проверяет\n\n| комплектность | заявки |\nв течение — пяти рабочих дней."
    )
    assert np.array_equal(signature(plain).values, signature(marked).values)


def test_word_order_matters():
    words = "один два три четыре пять шесть семь восемь девять десять".split()
    forward, backward = " ".join(words), " ".join(reversed(words))
    assert overlap(signature(forward), signature(backward)) == 0.0


def test_shingle_count_is_the_number_of_distinct_five_word_runs():
    assert signature("a b c d e f").shingles == 2
    assert signature("a b c d e a b c d e").shingles == 5  # "a b c d e" twice, once counted


def test_a_document_shorter_than_one_shingle_is_one_shingle_of_all_its_words():
    assert signature("Протокол совещания").shingles == 1
    assert overlap(signature("Протокол совещания"), signature("протокол  Совещания")) == 1.0
    assert overlap(signature("Протокол совещания"), signature("Протокол собрания")) == 0.0


def test_a_document_without_words_has_no_shingles():
    assert signature("").shingles == 0
    assert signature("— | --- | ***").shingles == 0


# Pinned from the first release of signatures. They are stored in an index and compared
# with signatures computed later, by other processes and other releases, so a change to
# the tokenizer, the shingle hash or the permutation seeds silently breaks every existing
# index. If this fails, the change is the bug.
PINNED_TEXT = "Отдел закупок проверяет комплектность заявки в течение пяти рабочих дней."
PINNED_HEAD = [
    64764032,
    296901094,
    768552816,
    108156892,
    206474400,
    445784176,
    148315169,
    1245163494,
]
PINNED_SHA256 = "26921a6947ea2a82d2fad0fe2a82cf323d22f8f6a8731bd6cbd4719e397da128"


def test_signatures_are_pinned_across_releases():
    sig = signature(PINNED_TEXT)
    assert sig.values.dtype == np.uint32 and sig.values.shape == (NUM_PERM,)
    assert sig.shingles == 6
    assert sig.values[:8].tolist() == PINNED_HEAD
    assert hashlib.sha256(sig.values.astype("<u4").tobytes()).hexdigest() == PINNED_SHA256


@pytest.mark.parametrize("seed", ["0", "1", "4242"])
def test_signatures_do_not_depend_on_the_process(seed, tmp_path):
    """Python's own `hash` of a str changes with every process; nothing here may use it."""
    script = (
        "import json, sys; from minirag_mcp.lookalike import signature; "
        "print(json.dumps(signature(sys.argv[1]).values.tolist()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, REGULATION_V1],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": ""},
    ).stdout
    assert json.loads(out) == signature(REGULATION_V1).values.tolist()


def _matrix(rows: list[int]) -> np.ndarray:
    """Row 0 is the probe; row k agrees with it in exactly rows[k] positions."""
    base = np.arange(NUM_PERM, dtype=np.uint32)
    out = [base]
    for agree in rows:
        row = base.copy()
        row[agree:] += 1_000_000
        out.append(row)
    return np.stack(out)


def test_the_threshold_is_inclusive_at_half_the_positions():
    index = SignatureIndex(
        ["/probe", "/half", "/below"], _matrix([NUM_PERM // 2, NUM_PERM // 2 - 1])
    )
    near = index.near("/probe")
    assert [n.source for n in near] == ["/half"]
    assert near[0].overlap == 0.5


def test_at_most_three_look_alikes_highest_overlap_first_never_itself():
    agree = [70, 120, 90, 128, 100]
    sources = ["/probe", "/a", "/b", "/c", "/d", "/e"]
    index = SignatureIndex(sources, _matrix(agree))
    near = index.near("/probe")
    assert len(near) == MAX_LOOK_ALIKES
    assert [n.source for n in near] == ["/d", "/b", "/e"]
    assert [n.overlap for n in near] == [1.0, 120 / NUM_PERM, 100 / NUM_PERM]


def test_equal_overlaps_are_ordered_by_source():
    index = SignatureIndex(["/probe", "/z", "/a"], _matrix([100, 100]))
    assert [n.source for n in index.near("/probe")] == ["/a", "/z"]


def test_a_source_the_index_does_not_hold_has_no_look_alikes():
    index = SignatureIndex(["/probe", "/a"], _matrix([128]))
    assert index.near("/nowhere") == []
    assert SignatureIndex([], np.zeros((0, NUM_PERM), dtype=np.uint32)).near("/a") == []
