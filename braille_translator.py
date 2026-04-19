"""
braille_translator.py

UEB (Unified English Braille) Grade 2 translator.

Bit mapping — index in the 6-character string:
  0 = Dot 1 (Top Left)      3 = Dot 4 (Top Right)
  1 = Dot 2 (Middle Left)   4 = Dot 5 (Middle Right)
  2 = Dot 3 (Bottom Left)   5 = Dot 6 (Bottom Right)

Example: 'c' = dots 1,4  →  "100100"
         'a' = dot 1     →  "100000"
"""

import re

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

ALPHABET: dict[str, str] = {
    'a': '100000', 'b': '110000', 'c': '100100', 'd': '100110',
    'e': '100010', 'f': '110100', 'g': '110110', 'h': '110010',
    'i': '010100', 'j': '010110', 'k': '101000', 'l': '111000',
    'm': '101100', 'n': '101110', 'o': '101010', 'p': '111100',
    'q': '111110', 'r': '111010', 's': '011100', 't': '011110',
    'u': '101001', 'v': '111001', 'w': '010111', 'x': '101101',
    'y': '101111', 'z': '101011',
}

NUMBERS: dict[str, str] = {
    '1': '100000', '2': '110000', '3': '100100', '4': '100110',
    '5': '100010', '6': '110100', '7': '110110', '8': '110010',
    '9': '010100', '0': '010110',
}

PUNCTUATION: dict[str, str] = {
    '.': '010011',   # dots 2,5,6
    ',': '010000',   # dot 2
    '?': '011001',   # dots 2,3,6
    '!': '011010',   # dots 2,3,5
    "'": '001000',   # dot 3
    '-': '001001',   # dots 3,6
    ':': '010010',   # dots 2,5
    ';': '011000',   # dots 2,3
}

NUMBER_INDICATOR = '001111'   # dots 3,4,5,6
SPACE            = '000000'

# Strong wordsigns — the entire word maps to one cell
STRONG_WORDSIGNS: dict[str, str] = {
    'and':  '111101',   # dots 1,2,3,4,6
    'for':  '111111',   # dots 1,2,3,4,5,6
    'of':   '111011',   # dots 1,2,3,5,6
    'the':  '011101',   # dots 2,3,4,6
    'with': '011111',   # dots 2,3,4,5,6
}

# Alphabetic wordsigns — word maps to its letter's cell
ALPHABETIC_WORDSIGNS: dict[str, str] = {
    'but':       'b', 'can':       'c', 'do':        'd',
    'every':     'e', 'from':      'f', 'go':        'g',
    'have':      'h', 'just':      'j', 'knowledge': 'k',
    'like':      'l', 'more':      'm', 'not':       'n',
    'people':    'p', 'quite':     'q', 'rather':    'r',
    'so':        's', 'that':      't', 'us':        'u',
    'very':      'v', 'will':      'w', 'it':        'x',
    'you':       'y', 'as':        'z',
}

_PUNCT_SET = set(PUNCTUATION)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_punct(token: str) -> tuple[str, str, str]:
    """Split 'token' into (leading_punctuation, core_word, trailing_punctuation)."""
    i = 0
    while i < len(token) and token[i] in _PUNCT_SET:
        i += 1
    j = len(token)
    while j > i and token[j - 1] in _PUNCT_SET:
        j -= 1
    return token[:i], token[i:j], token[j:]


def _chars_to_cells(text: str) -> list[tuple[str, str]]:
    """Convert a string character by character to (label, cell) pairs."""
    cells: list[tuple[str, str]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            cells.append(('[#]', NUMBER_INDICATOR))
            while i < len(text) and text[i].isdigit():
                cells.append((f'[{text[i]}]', NUMBERS[text[i]]))
                i += 1
        elif ch in ALPHABET:
            cells.append((f'[{ch}]', ALPHABET[ch]))
            i += 1
        elif ch in PUNCTUATION:
            cells.append((f'[{ch}]', PUNCTUATION[ch]))
            i += 1
        else:
            i += 1  # skip unknown characters silently
    return cells


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_to_braille(text: str) -> dict:
    """
    Translate plain text to UEB Grade 2 Braille.

    Returns:
        {
            "payload":   "100000,000000,011101,...",   # comma-separated 6-bit cells
            "debug_log": ["[a] -> 100000", "[space] -> 000000", ...]
        }
    """
    text = text.lower().strip()
    cells: list[tuple[str, str]] = []

    tokens = re.split(r'(\s+)', text)

    for token in tokens:
        if not token:
            continue

        if token.isspace():
            cells.append(('[space]', SPACE))
            continue

        lead, core, tail = _split_punct(token)

        for ch in lead:
            cells.append((f'[{ch}]', PUNCTUATION[ch]))

        if core in STRONG_WORDSIGNS:
            cells.append((f'[{core}] (wordsign)', STRONG_WORDSIGNS[core]))
        elif core in ALPHABETIC_WORDSIGNS:
            letter = ALPHABETIC_WORDSIGNS[core]
            cells.append((f'[{core}] (wordsign)', ALPHABET[letter]))
        elif core:
            cells.extend(_chars_to_cells(core))

        for ch in tail:
            cells.append((f'[{ch}]', PUNCTUATION[ch]))

    payload   = ','.join(cell for _, cell in cells)
    debug_log = [f'{label} -> {cell}' for label, cell in cells]

    return {'payload': payload, 'debug_log': debug_log}
