"""Reference solution — the creator-side benchmark for Flesch reading-ease.

A faithful implementation of the pinned spec. Scores 1.0, which is what proves
the spec in the overview is actually implementable as written.
"""

VOWELS = "aeiouy"
SENTENCE_ENDERS = ".!?"


def count_sentences(text):
    runs, in_run = 0, False
    for ch in text:
        if ch in SENTENCE_ENDERS:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs if runs else 1


def split_words(text):
    out = []
    for raw in text.split():
        start, end = 0, len(raw)
        while start < end and not raw[start].isalnum():
            start += 1
        while end > start and not raw[end - 1].isalnum():
            end -= 1
        if raw[start:end]:
            out.append(raw[start:end])
    return out


def count_syllables(word):
    lowered = word.lower()
    runs, in_run = 0, False
    for ch in lowered:
        if ch in VOWELS:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    if lowered.endswith("e") and runs > 1:
        runs -= 1
    return max(runs, 1)


class Agent:
    def __init__(self):
        pass

    def flesch_reading_ease(self, text):
        words = split_words(text)
        if not words:
            raise ValueError("flesch_reading_ease() requires at least one word")
        sentences = count_sentences(text)
        syllables = sum(count_syllables(w) for w in words)
        return (206.835
                - 1.015 * (len(words) / sentences)
                - 84.6 * (syllables / len(words)))
