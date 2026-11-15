"""Rule-based romanisation (ISO 15919) of Indic-script text.

bodhan-ai/indic-transcribe-core only emits native script, so `mode=romanised`
and `/v1/audio/translations` are completed here, after decoding. Brahmic scripts
are covered; Perso-Arabic, Ol Chiki and Meetei Mayek text is returned unchanged.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

try:
    from indic_transliteration import sanscript
except ImportError:  # optional dependency
    sanscript = None

# (first code point, last code point, sanscript scheme attribute)
_BLOCKS = (
    (0x0900, 0x097F, "DEVANAGARI"), (0x0980, 0x09FF, "BENGALI"), (0x0A00, 0x0A7F, "GURMUKHI"),
    (0x0A80, 0x0AFF, "GUJARATI"), (0x0B00, 0x0B7F, "ORIYA"), (0x0B80, 0x0BFF, "TAMIL"),
    (0x0C00, 0x0C7F, "TELUGU"), (0x0C80, 0x0CFF, "KANNADA"), (0x0D00, 0x0D7F, "MALAYALAM"),
)

_TA_VOWELS = {"அ": "a", "ஆ": "ā", "இ": "i", "ஈ": "ī", "உ": "u", "ஊ": "ū", "எ": "e", "ஏ": "ē",
              "ஐ": "ai", "ஒ": "o", "ஓ": "ō", "ஔ": "au", "ஃ": "ḵ"}
_TA_SIGNS = {"ா": "ā", "ி": "i", "ீ": "ī", "ு": "u", "ூ": "ū", "ெ": "e", "ே": "ē",
             "ை": "ai", "ொ": "o", "ோ": "ō", "ௌ": "au"}
_TA_CONSONANTS = {"க": "k", "ங": "ṅ", "ச": "c", "ஞ": "ñ", "ட": "ṭ", "ண": "ṇ", "த": "t", "ந": "n",
                  "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "ḻ", "ள": "ḷ",
                  "ற": "ṟ", "ன": "ṉ", "ஜ": "j", "ஷ": "ṣ", "ஸ": "s", "ஹ": "h", "ஶ": "ś"}
_TA_VIRAMA = "்"


def _tamil(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in _TA_CONSONANTS:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == _TA_VIRAMA:
                out.append(_TA_CONSONANTS[ch]); i += 2
            elif nxt in _TA_SIGNS:
                out.append(_TA_CONSONANTS[ch] + _TA_SIGNS[nxt]); i += 2
            else:
                out.append(_TA_CONSONANTS[ch] + "a"); i += 1
        else:
            out.append(_TA_VOWELS.get(ch, ch)); i += 1
    return "".join(out)


def _script_of(word: str) -> Optional[str]:
    for ch in word:
        cp = ord(ch)
        for lo, hi, name in _BLOCKS:
            if lo <= cp <= hi:
                return name
    return None


def romanise(text: str) -> Tuple[str, bool]:
    """Return (romanised text, changed). Words in unsupported scripts pass through."""
    changed = False
    parts = re.split(r"(\s+)", text)
    for i, word in enumerate(parts):
        script = _script_of(word)
        if script is None:
            continue
        if script == "TAMIL":
            parts[i] = _tamil(word)
        elif sanscript is not None:
            parts[i] = sanscript.transliterate(word, getattr(sanscript, script), sanscript.ISO)
        else:
            continue
        changed = True
    return "".join(parts), changed
