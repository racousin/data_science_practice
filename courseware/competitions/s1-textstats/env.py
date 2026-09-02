"""Session 1 — `textstats` correctness (flex_v1).

Lab 1 asks for three functions in `src/textstats/core.py`. This competition
grades exactly those three, on texts the agent has never seen.

The agent exposes them as methods with the Lab 1 signatures:

    class Agent:
        def word_count(self, text): ...
        def char_frequencies(self, text): ...
        def longest_word(self, text): ...

Score is the fraction of individual checks that match the reference exactly —
three checks per text, `len(CASES) * 3` in total.

Two deliberate omissions, both documented in the competition overview:

* The error path from Lab 1 Part C (`longest_word("")` must raise `ValueError`)
  is graded by your own pytest suite, not here. The platform latches a raised
  exception as an agent crash, so a competition that *required* a raise would
  mark every correct submission as broken. Every text sent here is non-empty.
* The first failed call latches the agent channel, so calls after it
  short-circuit. Partial credit is therefore "everything that passed before the
  first crash", and `info_message` names the case that crashed.
"""


# --- reference implementation (the spec, in code) --------------------------- #
#
# Tokens are whitespace-separated and keep their punctuation: "end." is one
# 4-character token. Characters are counted case-folded with whitespace
# dropped; punctuation and digits count. `longest_word` breaks ties by taking
# the FIRST token of maximal length.

def _word_count(text):
    return len(text.split())


def _char_frequencies(text):
    counts = {}
    for ch in text.lower():
        if ch.isspace():
            continue
        counts[ch] = counts.get(ch, 0) + 1
    return counts


def _longest_word(text):
    tokens = text.split()
    if not tokens:
        raise ValueError("longest_word() requires at least one token")
    best = tokens[0]
    for token in tokens[1:]:
        if len(token) > len(best):
            best = token
    return best


# --- hidden evaluation set -------------------------------------------------- #
#
# env.py is creator-only storage — competitors never see this list. Cases are
# ordered easy -> awkward so a crash message points at a specific behaviour.

CASES = [
    "the quick brown fox jumps over the lazy dog",
    "Hello",
    "Hello world",
    "  leading and trailing whitespace   ",
    "tabs\tand\nnewlines\tcount as whitespace",
    "Repeated repeated REPEATED words words",
    "Punctuation, of course; belongs to the token it ends.",
    "MiXeD CaSe ShOuLd FoLd To LoWeR",
    "digits 1234567890 are characters too",
    "a bb ccc dddd ccc bb a",
    "tie breaking aaaa bbbb between equals",
    "hyphenated-words stay one single token",
    "under_scores and dots.in.names are not separators",
    "naive cafe resume vs naive cafe resume",
    "unicode accents: elephant edifice etudiant",
    "emoji do not break character counting",
    "one",
    "x",
    "symbols !@#$%^&*() are counted individually",
    "the end of the hidden set is here",
]

_CHECKS = ("word_count", "char_frequencies", "longest_word")

_REFERENCE = {
    "word_count": _word_count,
    "char_frequencies": _char_frequencies,
    "longest_word": _longest_word,
}


def _matches(method, got, expected):
    """Exact comparison, with the small amount of slack JSON transport needs.

    `char_frequencies` may legitimately come back as a `Counter`; the wire
    turns it into a plain object either way, so a dict compare is enough.
    `word_count` may arrive as 3.0 rather than 3 — accept an integral float,
    reject a boolean (True == 1 in Python and that is not a word count).
    """
    if method == "word_count":
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return False
        return float(got) == float(expected)
    if method == "char_frequencies":
        if not isinstance(got, dict):
            return False
        if set(got.keys()) != set(expected.keys()):
            return False
        for key, value in expected.items():
            other = got[key]
            if isinstance(other, bool) or not isinstance(other, (int, float)):
                return False
            if float(other) != float(value):
                return False
        return True
    if method == "longest_word":
        return isinstance(got, str) and got == expected
    raise AssertionError(f"unknown check {method!r}")


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        # Precompute the answers once; the loop below stays a pure comparison.
        self.expected = [
            {name: fn(text) for name, fn in _REFERENCE.items()}
            for text in CASES
        ]
        self.total_checks = len(CASES) * len(_CHECKS)

    def evaluate(self, agents, agent_infos):
        return {"agent_results": [
            self._score_one(index, proxy) for index, proxy in enumerate(agents)
        ]}

    def _score_one(self, index, proxy):
        per_method = {name: 0 for name in _CHECKS}
        first_error = None

        for case_index, text in enumerate(CASES):
            for name in _CHECKS:
                got = proxy.call(name, text, catch_errors=True)
                if proxy.last_error is not None:
                    if first_error is None:
                        first_error = (
                            f"case #{case_index} `{name}` failed: {proxy.last_error}"
                        )
                    continue
                if _matches(name, got, self.expected[case_index][name]):
                    per_method[name] += 1

        passed = sum(per_method.values())
        pass_rate = passed / self.total_checks

        if first_error is not None:
            info = (
                f"{passed}/{self.total_checks} checks passed before the agent "
                f"crashed. {first_error}"
            )
        else:
            info = f"{passed}/{self.total_checks} checks passed."

        return {
            "agent_index": index,
            "score": round(pass_rate, 6),
            "steps": self.total_checks,
            "info_message": info,
            "metrics_detail": {
                "pass_rate": round(pass_rate, 6),
                "word_count_ok": round(per_method["word_count"] / len(CASES), 6),
                "char_frequencies_ok": round(
                    per_method["char_frequencies"] / len(CASES), 6),
                "longest_word_ok": round(
                    per_method["longest_word"] / len(CASES), 6),
            },
        }
