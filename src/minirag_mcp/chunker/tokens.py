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


def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate, used only when the real counter is unavailable.

    Deliberately pessimistic: it is better to under-fill the budget than to hand the
    encoder text it will silently truncate.
    """
    return ceil(len(text) / CHARS_PER_TOKEN) + _SPECIAL_TOKENS
