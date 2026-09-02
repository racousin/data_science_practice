"""Not shipped — a deliberately broken agent used to check the scoring path."""


class Agent:
    def __init__(self):
        self.calls = 0

    def word_count(self, text):
        self.calls += 1
        if self.calls > 8:          # crash partway through the hidden set
            raise ValueError("boom on a perfectly valid text")
        return len(text.split())

    def char_frequencies(self, text):
        return {}                    # wrong, but does not raise

    def longest_word(self, text):
        return max(text.split(), key=len)   # wrong tie-breaking
