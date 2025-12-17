"""Thai text utilities for grapheme decomposition and Unicode handling."""

import re
from typing import Dict, List

# --- Unicode Ranges ---
THAI_LEADING_VOWELS = set('\u0E40\u0E41\u0E42\u0E43\u0E44') # เ แ โ ไ ใ
THAI_UPPER_VOWELS = set('\u0E31\u0E34\u0E35\u0E36\u0E37\u0E47') # อั อิ อี อึ อือ ไม้ไต่คู้
THAI_LOWER_CHARS = set('\u0E38\u0E39\u0E3A') # อุ อู พินทุ
THAI_TONE_MARKS = set('\u0E48\u0E49\u0E4A\u0E4B') # เอก โท ตรี จัตวา
THAI_UPPER_DIACRITICS = set('\u0E4C\u0E4D\u0E4E') # การันต์, นิคหิต, ยามักการ
THAI_TRAILING_VOWELS = set('\u0E30\u0E32\u0E33\u0E45') # อะ อา อำ ฯ
THAI_SPECIAL_CHARS = set('\u0E46\u0E2F')

THAI_UPPER_CHARS = THAI_UPPER_VOWELS | THAI_TONE_MARKS | THAI_UPPER_DIACRITICS

SARA_AM = '\u0E33'
NIKHAHIT = '\u0E4D'
SARA_AA = '\u0E32'

# --- Glyph Names for Vector Analysis ---
# ใช้ตัวเล็กทั้งหมดเพื่อ Case Insensitive Matching
NAME_NIKHAHIT = ['uni0e4d', 'nikhahit', 'bindu', 'afii59757']
NAME_SARA_AA = ['uni0e32', 'sara_aa', 'afii59753']
NAME_SARA_AM = ['uni0e33', 'sara_am', 'saraam', 'afii59758']
NAME_TONES = ['uni0e48', 'uni0e49', 'uni0e4a', 'uni0e4b', 'mai_ek', 'mai_tho', 'mai_tri', 'mai_jattawa']


def decompose_thai_grapheme(grapheme: str) -> Dict:
    """Decompose Thai grapheme into components. (Legacy)"""
    if not grapheme:
        return {
            'base': '', 'leading': '', 'upper_vowel': '', 'upper_tone': '',
            'upper_diacritic': '', 'lower': '', 'trailing': '', 'is_sara_am': False
        }

    if SARA_AM in grapheme:
        base = grapheme.replace(SARA_AM, '')
        leading, upper_vowel, upper_tone, upper_diacritic_extra, lower, base_only = '', '', '', '', '', ''

        for char in base:
            if char in THAI_LEADING_VOWELS: leading += char
            elif char in THAI_UPPER_VOWELS: upper_vowel += char
            elif char in THAI_TONE_MARKS: upper_tone += char
            elif char in THAI_UPPER_DIACRITICS: upper_diacritic_extra += char
            elif char in THAI_LOWER_CHARS: lower += char
            else: base_only += char

        return {
            'base': base_only,
            'leading': leading,
            'upper_vowel': upper_vowel,
            'upper_tone': upper_tone,
            'upper_diacritic': NIKHAHIT + upper_diacritic_extra,
            'lower': lower,
            'trailing': SARA_AA,
            'is_sara_am': True
        }

    base, leading, upper_vowel, upper_tone, upper_diacritic, lower = '', '', '', '', '', ''

    for char in grapheme:
        if char in THAI_LEADING_VOWELS: leading += char
        elif char in THAI_UPPER_VOWELS: upper_vowel += char
        elif char in THAI_TONE_MARKS: upper_tone += char
        elif char in THAI_UPPER_DIACRITICS: upper_diacritic += char
        elif char in THAI_LOWER_CHARS: lower += char
        else: base += char

    return {
        'base': base,
        'leading': leading,
        'upper_vowel': upper_vowel,
        'upper_tone': upper_tone,
        'upper_diacritic': upper_diacritic,
        'lower': lower,
        'trailing': '',
        'is_sara_am': False
    }


def reorder_thai_combining(text: str) -> str:
    """Reorder Thai combining characters to proper sequence. (Legacy)"""
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        if char not in THAI_UPPER_CHARS and char not in THAI_LOWER_CHARS and char not in THAI_LEADING_VOWELS:
            base = char
            combining, upper_diacritics, tone_marks, trailing = [], [], [], []
            i += 1
            while i < len(text):
                c = text[i]
                if c in THAI_UPPER_DIACRITICS: upper_diacritics.append(c)
                elif c in THAI_TONE_MARKS: tone_marks.append(c)
                elif c in THAI_UPPER_VOWELS or c in THAI_LOWER_CHARS: combining.append(c)
                elif c == SARA_AA or c in THAI_LEADING_VOWELS: trailing.append(c)
                else: break
                i += 1
            result.append(base)
            result.extend(combining)
            result.extend(upper_diacritics)
            result.extend(tone_marks)
            result.extend(trailing)
        else:
            result.append(char)
            i += 1
    return ''.join(result)


def split_grapheme_clusters(text: str) -> List[str]:
    """Split text into Thai grapheme clusters. (Legacy)"""
    th_pattern = r'[\u0E00-\u0E7F][\u0E31\u0E33\u0E34-\u0E3A\u0E47-\u0E4E]*'
    clusters = []
    pos = 0
    for match in re.finditer(th_pattern, text):
        if match.start() > pos: clusters.extend(list(text[pos:match.start()]))
        clusters.append(match.group())
        pos = match.end()
    if pos < len(text): clusters.extend(list(text[pos:]))
    return clusters


def has_upper_vowel(grapheme: str) -> bool:
    return any(char in THAI_UPPER_CHARS for char in grapheme)


def has_lower_vowel(grapheme: str) -> bool:
    return any(char in THAI_LOWER_CHARS for char in grapheme)


def normalize_grapheme(grapheme: str) -> str:
    normalized = grapheme.replace(SARA_AM, NIKHAHIT + SARA_AA)
    return reorder_thai_combining(normalized)


def contains_thai(text: str) -> bool:
    return any('\u0E00' <= c <= '\u0E7F' for c in text)