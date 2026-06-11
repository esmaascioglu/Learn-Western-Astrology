"""Chunking primitives: heading recognition, entity tagging, junk filtering.

These are the behaviours the per-document strategies depend on, isolated from
PDF I/O so they run fast and deterministically.
"""
from __future__ import annotations

from app.ingestion.chunk import (
    HEAD_RE,
    HOUSE_RE,
    _classify_font_heading,
    _hand_meta,
    _is_useful,
    _tag,
    _windows,
)


# --- heading recognition + typed metadata --------------------------------
def test_aspect_heading_is_tagged_as_transit_aspect():
    assert HEAD_RE.match("Sun Conjunct Moon")
    meta = _hand_meta("Sun Conjunct Moon")
    assert meta["indicator_type"] == "transit_aspect"
    assert meta["transiting"] == "sun"
    assert meta["aspect"] == "conjunct"
    assert meta["natal"] == "moon"
    assert meta["context"] == "transit"


def test_aspect_to_an_angle_is_tagged_as_angle_aspect():
    # Aspects to the Ascendant/Midheaven are angle aspects, not planet aspects.
    assert _hand_meta("Sun Conjunct Ascendant")["indicator_type"] == "angle_aspect"


def test_house_ingress_heading_resolves_ordinal_to_number():
    assert HOUSE_RE.match("Saturn in the Seventh House")
    meta = _hand_meta("Saturn in the Seventh House")
    assert meta["indicator_type"] == "house_ingress"
    assert meta["transiting"] == "saturn"
    assert meta["house"] == 7


def test_non_heading_falls_back_to_bare_transit_context():
    assert _hand_meta("just some prose")["context"] == "transit"


def test_font_heading_classification():
    assert _classify_font_heading("Venus in Gemini", None)["indicator_type"] == "planet_in_sign"
    assert _classify_font_heading("Aries Ascendant", None)["indicator_type"] == "ascendant"
    assert _classify_font_heading("Taurus", None)["indicator_type"] == "sign_chapter"
    # A generic subsection inherits the current sign chapter as breadcrumb context.
    sub = _classify_font_heading("DEPENDABILITY", "taurus")
    assert sub["indicator_type"] == "sign_section"
    assert sub["sign"] == "taurus"


# --- entity tagging -------------------------------------------------------
def test_tag_extracts_sorted_unique_entities():
    tags = _tag("Venus square Mars while the Moon is in Gemini")
    assert tags["planets"] == ["mars", "moon", "venus"]
    assert tags["signs"] == ["gemini"]
    assert tags["aspects"] == ["square"]


def test_tag_omits_absent_categories():
    assert _tag("a passage with no astrological entities") == {}


# --- junk filtering -------------------------------------------------------
def test_prose_is_useful():
    assert _is_useful("The Sun represents your core identity and ego in the natal chart always")


def test_uppercase_code_table_is_dropped():
    # Ephemeris/index rows are mostly uppercase codes and numbers -> not useful.
    assert not _is_useful("PIS ARI TAU GEM CAN LEO VIR LIB SCO SAG CAP AQU")


def test_too_short_is_dropped():
    assert not _is_useful("Sun Moon Mars")


# --- window packing -------------------------------------------------------
def test_windows_prepend_heading_and_track_pages():
    body = [(4, "alpha beta"), (5, "gamma delta")]
    windows = _windows(body, target=1000, overlap=0, heading="HEAD")
    assert len(windows) == 1
    content, page_start, page_end = windows[0]
    assert content.startswith("HEAD")
    assert "alpha" in content and "delta" in content
    assert (page_start, page_end) == (4, 5)
