"""Query focusing — strip generic astrology *framing* words before lexical search.


IMPORTANT: FRAME contains ONLY generic question/framing vocabulary. Planet, sign,
house, and aspect names, and trait words, are NOT here — those are the content we
want to keep. Applied to the FTS + SPLADE (lexical) legs; the dense leg keeps the
full natural-language question (it handles paraphrase/context, not keywords).
"""
from __future__ import annotations

import re

# Generic astrology framing + English question scaffolding. These words appear
# across the whole corpus (or carry no astrological content), so matching on them
# is pure noise. Deliberately EXCLUDES: planets, signs, houses, ordinals,
# aspect names (conjunct/square/trine/…), "transit", "house", and trait words.
FRAME = {
    # question words / generic verbs
    "which", "what", "who", "whose", "whom", "when", "where", "why", "how",
    "make", "makes", "making", "made", "cause", "causes", "caused", "give",
    "gives", "given", "mean", "means", "meaning", "indicate", "indicates",
    "show", "shows", "represent", "represents", "describe", "tell", "explain",
    # pronouns / determiners / particles
    "you", "your", "yours", "my", "mine", "me", "i", "we", "our", "us",
    "a", "an", "the", "in", "of", "to", "do", "does", "did", "is", "are",
    "am", "be", "been", "being", "on", "for", "with", "that", "this", "these",
    "those", "and", "or", "but", "if", "as", "at", "by", "from", "about",
    "person", "people", "someone", "somebody", "one", "their", "them", "they",
    "he", "she", "his", "her", "him", "its", "it",
    # generic astrology scaffolding (apply to every chunk → no discrimination)
    "combination", "combinations", "placement", "placements",
    "position", "positions", "aspect", "aspects",
    "natal", "birth", "chart", "charts", "horoscope", "horoscopes",
    "astrology", "astrological", "zodiac", "sign", "signs",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


def focus_query(question: str) -> str:
    """Return the question with generic framing words removed.

    Falls back to the original question if stripping leaves nothing (e.g. the
    query was entirely framing words)."""
    kept = [w for w in _WORD_RE.findall(question.lower()) if w not in FRAME]
    return " ".join(kept) if kept else question
