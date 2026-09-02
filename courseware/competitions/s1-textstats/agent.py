"""Reference solution — the creator-side benchmark for `textstats` correctness.

Run as the competition's benchmark agent: a correct implementation must score
exactly 1.0, which is what proves the env grades what it claims to grade.
"""


class Agent:
    def __init__(self):
        pass

    def word_count(self, text):
        """Number of whitespace-separated tokens in `text`."""
        return len(text.split())

    def char_frequencies(self, text):
        """Count of each character, ignoring whitespace and case."""
        counts = {}
        for ch in text.lower():
            if ch.isspace():
                continue
            counts[ch] = counts.get(ch, 0) + 1
        return counts

    def longest_word(self, text):
        """The longest token; ties go to the first. Raises on empty input."""
        tokens = text.split()
        if not tokens:
            raise ValueError("longest_word() requires at least one token")
        best = tokens[0]
        for token in tokens[1:]:
            if len(token) > len(best):
                best = token
        return best
