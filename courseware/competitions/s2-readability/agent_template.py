"""Session 2 starter — the readability module from Lab 2, on the leaderboard.

Upload `readability.py` from your package next to this file and import it (the
agent directory is on `sys.path`), or implement it here directly.

The competition overview pins the sentence, word and syllable rules exactly.
Feed that specification to your agent verbatim — vague prompts produce vague
syllable counters, and a syllable counter that is off by one on "queueing" is
off by several points of reading ease.
"""


class Agent:
    def __init__(self):
        pass

    def flesch_reading_ease(self, text):
        """206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words).

        See the overview for how `words`, `sentences` and `syllables` are
        defined. Every text you are sent has at least one word.
        """
        raise NotImplementedError
