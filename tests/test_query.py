"""Query focusing: framing words are stripped, discriminating content is kept."""
from __future__ import annotations

from app.retrieval.query import focus_query


def test_trait_question_reduces_to_the_trait():
    # The whole point: a natural-language trait question collapses to its one
    # discriminating word, which is what the lexical legs should match on.
    assert focus_query("Which combinations in a natal chart make you stubborn?") == "stubborn"
    assert focus_query("What makes someone confident in their natal chart?") == "confident"


def test_structural_query_keeps_planet_and_sign():
    # Planet/sign names are content, never framing — they must survive.
    assert focus_query("What does Venus in Gemini mean in a natal chart?") == "venus gemini"


def test_house_and_transit_words_are_preserved():
    # "house", "transit", "seventh", "through" carry meaning; only "the" is framing.
    assert focus_query("Saturn transit through the seventh house") == \
        "saturn transit through seventh house"


def test_all_framing_falls_back_to_original():
    # If stripping leaves nothing, return the original (never an empty query).
    q = "What is it?"
    assert focus_query(q) == q


def test_is_case_insensitive_and_drops_punctuation():
    assert focus_query("STUBBORN?!") == "stubborn"
