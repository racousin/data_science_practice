"""Session 1 starter — wire your Lab 1 package up to the leaderboard.

Two ways to fill this in:

1.  Upload `core.py` from your package next to this file and import it. The
    agent directory is on `sys.path`, so a plain `from core import ...` works:

        from core import word_count, char_frequencies, longest_word

        class Agent:
            def word_count(self, text):
                return word_count(text)
            ...

2.  Or paste the three function bodies straight into the methods below.

Read the competition overview first — it pins the tokenisation and tie-breaking
rules the reference uses. Every text you are sent is non-empty, so the
`ValueError` path from Lab 1 Part C is never exercised here; keep it anyway,
your own tests check it.
"""


class Agent:
    def __init__(self):
        pass

    def word_count(self, text):
        """Number of whitespace-separated tokens in `text`."""
        raise NotImplementedError

    def char_frequencies(self, text):
        """Count of each character, ignoring whitespace and case."""
        raise NotImplementedError

    def longest_word(self, text):
        """The longest token. Ties go to the first one in the text."""
        raise NotImplementedError
