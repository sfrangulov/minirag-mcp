"""Look-alike documents: how much text two indexed documents share.

A document's signature is a MinHash over the set of its five-word shingles; the share of
positions at which two signatures agree estimates the Jaccard similarity of the two sets.
The estimate is a measurement of shared text and nothing more. On a real corpus a pair
at 0.7 or above was a version of one document in every case sampled, while 0.5 to 0.7
was mostly siblings filled in from one template, and which of two versions is the newer
could not be told reliably from text or metadata. So a look-alike is reported, never
classified or ordered by age.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NUM_PERM = 128
SHINGLE_WORDS = 5
THRESHOLD = 0.5
MAX_LOOK_ALIKES = 3
# Agreeing positions a pair needs; counted in integers so 0.5 is not left to rounding.
_AGREE = math.ceil(THRESHOLD * NUM_PERM)
# The largest prime below 2**32. With a, b and a shingle hash all below 2**32, a*x + b
# stays below 2**64, so the permutations are exact in uint64 and never wrap.
_PRIME = 4_294_967_291
_WORD = re.compile(r"\w+")
# Shingles hashed per step: memory is NUM_PERM * _BATCH * 8 bytes.
_BATCH = 4096
_MIX = np.uint64(0x9E3779B97F4A7C15)
_FINAL = np.uint64(0xFF51AFD7ED558CCD)


def _seeded(label: str) -> int:
    return int.from_bytes(hashlib.blake2b(label.encode(), digest_size=8).digest(), "big")


# From fixed labels rather than a random generator: signatures are stored, and compared
# with ones computed by other processes and later releases, so these must never change.
_A = np.array(
    [_seeded(f"minirag-minhash-a-{i}") % (_PRIME - 1) + 1 for i in range(NUM_PERM)],
    dtype=np.uint64,
)
_B = np.array(
    [_seeded(f"minirag-minhash-b-{i}") % _PRIME for i in range(NUM_PERM)], dtype=np.uint64
)


@dataclass(frozen=True)
class Signature:
    values: np.ndarray  # uint32[NUM_PERM]
    # Distinct shingles; a document with none matches nothing, not even another empty one.
    shingles: int


def words(text: str) -> list[str]:
    """NFC first: a decomposed й is и plus a combining mark, which `\\w` does not match."""
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def _shingle_hashes(tokens: Sequence[str]) -> np.ndarray:
    """Distinct 32-bit hashes of the five-word runs; fewer words make one run of them all."""
    if not tokens:
        return np.zeros(0, dtype=np.uint64)
    vocab: dict[str, int] = {}
    codes = np.array([vocab.setdefault(t, len(vocab)) for t in tokens], dtype=np.int64)
    by_token = np.array([_seeded(t) for t in vocab], dtype=np.uint64)[codes]
    width = min(SHINGLE_WORDS, len(tokens))
    count = len(tokens) - width + 1
    h = by_token[:count].copy()
    for j in range(1, width):
        h = h * _MIX + by_token[j : j + count]
    h ^= h >> np.uint64(33)
    h *= _FINAL
    h ^= h >> np.uint64(33)
    return np.unique(h >> np.uint64(32))


def signature(text: str) -> Signature:
    hashes = _shingle_hashes(words(text))
    out = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, hashes.size, _BATCH):
        part = hashes[start : start + _BATCH]
        permuted = (_A[:, None] * part[None, :] + _B[:, None]) % np.uint64(_PRIME)
        np.minimum(out, permuted.min(axis=1), out=out)
    if hashes.size == 0:
        out[:] = np.iinfo(np.uint32).max
    return Signature(out.astype(np.uint32), int(hashes.size))


def overlap(a: Signature, b: Signature) -> float:
    """Estimated Jaccard similarity of two documents' shingle sets, 0..1."""
    if a.shingles == 0 or b.shingles == 0:
        return 0.0
    return int(np.count_nonzero(a.values == b.values)) / NUM_PERM


@dataclass(frozen=True)
class LookAlike:
    source: str
    overlap: float
    doc_date: str = ""
    doc_date_source: str = ""


class SignatureIndex:
    """Every comparable signature of an index, one row per source."""

    def __init__(self, sources: Sequence[str], matrix: np.ndarray):
        self.sources = list(sources)
        self.matrix = matrix
        self._row = {s: i for i, s in enumerate(self.sources)}

    def near(self, source: str) -> list[LookAlike]:
        """The documents sharing at least THRESHOLD of `source`'s text, most first."""
        i = self._row.get(source)
        if i is None:
            return []
        agree = np.count_nonzero(self.matrix == self.matrix[i], axis=1)
        agree[i] = 0
        hits = np.flatnonzero(agree >= _AGREE)
        ranked = sorted(hits.tolist(), key=lambda j: (-int(agree[j]), self.sources[j]))
        return [
            LookAlike(self.sources[j], int(agree[j]) / NUM_PERM) for j in ranked[:MAX_LOOK_ALIKES]
        ]
