"""Degeneration checks, from the same author's document pipeline (MIT).

These are the guards that caught 236 prompt-echo documents and 105 looping pages in
production. A page flagged here is scored as degenerate, not dropped from the results.
"""

from __future__ import annotations

from collections import Counter

# A loop needs both: half the lines duplicates, and one line at least five times. A
# 29-row table is distinct rows, so it passes.
REPETITION_DUPLICATE_SHARE = 0.50
REPETITION_TOP_LINE_COUNT = 5
REPETITION_MIN_LINES = 8


def repetition_loop(text: str) -> tuple[str, int, int] | None:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < REPETITION_MIN_LINES:
        return None
    counts = Counter(lines)
    top_line, top_count = counts.most_common(1)[0]
    duplicate_share = 1 - (len(counts) / len(lines))
    if duplicate_share < REPETITION_DUPLICATE_SHARE or top_count < REPETITION_TOP_LINE_COUNT:
        return None
    return top_line, top_count, len(lines)


def prompt_echo_markers(prompt_text: str) -> tuple[str, ...]:
    """Distinctive headings of a long prompt (lines starting with #, at least 12 chars)."""
    markers = []
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip().rstrip(":").strip()
        if len(heading) >= 12 and any(c.isalpha() for c in heading):
            markers.append(heading.upper())
    return tuple(dict.fromkeys(markers))


def echo_detected(text: str, prompt: str) -> bool:
    """Did the model transcribe its instructions instead of the page?"""
    if not prompt or not text:
        return False
    markers = prompt_echo_markers(prompt)
    if markers:
        haystack = text.upper()
        return any(marker in haystack for marker in markers)
    opening = " ".join(prompt.split()[:6])
    return len(opening) >= 10 and opening.lower() in text.lower()


def degeneration(text: str, prompt: str, finish_reasons: list[str]) -> dict[str, bool]:
    return {
        "echo": echo_detected(text, prompt),
        "loop": repetition_loop(text) is not None,
        "truncated": any(reason == "length" for reason in finish_reasons),
        "empty": not text.strip(),
    }
