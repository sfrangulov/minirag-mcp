"""Token counting — the chunker's budget is measured in the model's own tokens.

The binding constraint is the encoder's trained sequence length (128 tokens for
`paraphrase-multilingual-MiniLM-L12-v2`), and that is a token limit, not a character
limit. A character budget silently over- or under-fills depending on the language
mix: measured over the 558-document corpus, prose tokenizes at ~3.3 characters per
token and markdown table rows at ~2.2, a 50% spread. So the chunker takes a
`CountTokens` callable and asks the real tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil

# text -> number of tokens the embedding model would produce for it
CountTokens = Callable[[str], int]

# Fallback ratio for `estimate_tokens`, deliberately below every ratio measured on the
# corpus so the estimate errs towards *more* tokens and therefore smaller chunks.
# Measured with the model's own tokenizer, truncation and padding disabled, over all
# 50,575 chunks of the live index: 3.45 chars/token overall, 3.32 median per document,
# 2.88 at the 5th percentile per document, and 2.2 for markdown table rows.
CHARS_PER_TOKEN = 2.5
# Every tokenizer in this family wraps a sequence in a start and an end token.
_SPECIAL_TOKENS = 2


def _standalone(ch: str) -> bool:
    """Characters this tokenizer family almost always gives a token of their own.

    A markdown delimiter row is the extreme case: `| --- | --- |` is 13 characters and
    10 tokens, four times denser than prose. Counting punctuation and digits one for
    one cuts the estimate's under-counts from 14.3% of corpus lines to 4.6%.
    """
    return ch.isdigit() or (not ch.isalnum() and not ch.isspace())


def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate, used only when the real counter is unavailable.

    Deliberately pessimistic — measured over 8,000 real corpus lines it comes out 1.40x
    the true count on average — because it is better to under-fill the budget than to
    hand the encoder text it will silently truncate. It is an estimate, not a bound: no
    character heuristic can be exact about `ax_ИсторияЗаменыТМЗКЗакупу`, and the worst
    line measured still lands 26% under. That is why it is only the fallback.
    """
    dense = sum(1 for ch in text if _standalone(ch))
    return ceil((len(text) - dense) / CHARS_PER_TOKEN) + dense + _SPECIAL_TOKENS
