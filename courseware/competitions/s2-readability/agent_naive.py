"""Not shipped — a plausible-but-sloppy implementation, to check the spec discriminates."""
import re


class Agent:
    def __init__(self):
        pass

    def flesch_reading_ease(self, text):
        sentences = max(len(re.findall(r"[.!?]", text)), 1)     # counts "?!" as 2
        words = text.split()                                     # keeps punctuation
        syllables = sum(max(len(re.findall(r"[aeiou]", w.lower())), 1) for w in words)
        return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
