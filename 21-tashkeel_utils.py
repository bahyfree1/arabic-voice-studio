# -*- coding: utf-8 -*-
"""Automatic Arabic diacritization (tashkeel) using the open-source Mishkal library.

The result is a strong first draft, not a guarantee: always let the user review
and fix the vocalized text before synthesis. That review step is built into the UI.
"""

_vocalizer = None


def _get_vocalizer():
    global _vocalizer
    if _vocalizer is None:
        import mishkal.tashkeel
        _vocalizer = mishkal.tashkeel.TashkeelClass()
    return _vocalizer


def vocalize(text: str) -> str:
    """Add tashkeel to Arabic text, line by line (empty lines are preserved)."""
    v = _get_vocalizer()
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        out_lines.append(v.tashkeel(stripped) if stripped else "")
    return "\n".join(out_lines)


if __name__ == "__main__":
    import sys
    src = sys.stdin.read()
    print(vocalize(src))
