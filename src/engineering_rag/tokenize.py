"""Simple, dependency-free tokenizer used by the BM25 index.

Lowercases the input and splits it into word-like tokens with a regex that
allows internal hyphens/underscores/slashes (so equipment tags like
``CP-100`` or units like ``m3/h`` survive as single tokens), then optionally
drops a small, fixed English stopword list.

Deliberately simple: no external NLP library, no stemming, no lemmatization.
That keeps indexing and evaluation fully reproducible with zero installation
footprint -- the whole retrieval half of this project has exactly one
runtime dependency: the Python standard library.
"""

from __future__ import annotations

import re

# Starts and ends with an alphanumeric run; internal hyphen/underscore/slash
# is allowed as long as it is followed by another alphanumeric run. This
# keeps "cp-100" and "m3/h" whole while still splitting on a trailing period
# or other punctuation.
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*")

# A small, fixed stopword list (English). Deliberately short: engineering
# text is full of short-but-meaningful tokens (unit symbols, tag fragments),
# so an aggressive stopword list would strip signal, not just noise.
STOPWORDS = frozenset(
    """
    a an the of to in on for and or is are was were be been being with
    as at by from this that these those it its into over under
    """.split()
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenize `text` into a list of lowercase word tokens.

    >>> tokenize("The Pump Model CP-100 delivers 250 m3/h.")
    ['pump', 'model', 'cp-100', 'delivers', '250', 'm3/h']
    """
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
