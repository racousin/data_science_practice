"""Session 2 — Flesch reading-ease (flex_v1).

Lab 2 has you drive a coding agent to implement the Flesch reading-ease score.
The formula is not the hard part; the *specification* is. Syllable counting has
no canonical answer, so this competition pins one — the rules below are the
whole contract, and they are reproduced verbatim in the competition overview.

    FRE = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

Pinned rules
------------
sentences  Count the maximal runs of characters drawn from `.!?`; "Wait... no!"
           is two sentences, not four. A text with no such punctuation counts
           as one sentence.
words      Split on whitespace, then strip leading and trailing characters that
           are not letters or digits. Tokens that become empty are dropped.
           "end." -> "end"; "--" -> dropped.
syllables  Lowercase the word and count the maximal runs of `aeiouy`. If the
           word ends in "e" and that count is greater than 1, subtract 1.
           The result is never less than 1.

The agent exposes one method:

    class Agent:
        def flesch_reading_ease(self, text): ...

Score is the fraction of hidden texts scored within `TOLERANCE` of the
reference. As in Session 1, the empty-input error path is graded by your own
tests, not here: the platform latches a raised exception as an agent crash, so
every text sent is non-empty and has at least one word.
"""

TOLERANCE = 1e-6

_VOWELS = "aeiouy"
_SENTENCE_ENDERS = ".!?"


# --- reference implementation (the spec, in code) --------------------------- #

def _count_sentences(text):
    runs = 0
    in_run = False
    for ch in text:
        if ch in _SENTENCE_ENDERS:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs if runs else 1


def _words(text):
    out = []
    for raw in text.split():
        start, end = 0, len(raw)
        while start < end and not raw[start].isalnum():
            start += 1
        while end > start and not raw[end - 1].isalnum():
            end -= 1
        token = raw[start:end]
        if token:
            out.append(token)
    return out


def _count_syllables(word):
    lowered = word.lower()
    runs = 0
    in_run = False
    for ch in lowered:
        if ch in _VOWELS:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    if lowered.endswith("e") and runs > 1:
        runs -= 1
    return max(runs, 1)


def _flesch_reading_ease(text):
    words = _words(text)
    if not words:
        raise ValueError("flesch_reading_ease() requires at least one word")
    sentences = _count_sentences(text)
    syllables = sum(_count_syllables(w) for w in words)
    return (206.835
            - 1.015 * (len(words) / sentences)
            - 84.6 * (syllables / len(words)))


# --- hidden evaluation set -------------------------------------------------- #

CASES = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "This sentence has no terminal punctuation",
    "Wait... what happened?! Nobody knows.",
    "Short. Sentences. Everywhere. Score. Goes. Up.",
    "Notwithstanding the aforementioned considerations, the committee "
    "unanimously determined that the proposal warranted comprehensive "
    "reevaluation.",
    "One",
    "e",
    "The value is 42 dollars and 17 cents.",
    "Hyphenated-words and under_scores keep their inner punctuation.",
    "-- --- ---- words survive stripping ---- --- --",
    "Some words end in silent e like time and place and rate.",
    "Rhythm myths fly by dryly.",
    "A B C D E F G",
    "Mr. Smith went to Washington. He stayed for three weeks.",
    "Programming requires patience, precision, and an unreasonable tolerance "
    "for ambiguity in specifications.",
    "Why? Because!",
    "queue queueing queued",
    "The reevaluation of the aforementioned methodology proved inconclusive",
    "Simple text, plain and clear, easy to read for everyone here.",
]


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        self.expected = [_flesch_reading_ease(text) for text in CASES]

    def evaluate(self, agents, agent_infos):
        return {"agent_results": [
            self._score_one(index, proxy) for index, proxy in enumerate(agents)
        ]}

    def _score_one(self, index, proxy):
        passed = 0
        errors = []
        first_error = None

        for case_index, text in enumerate(CASES):
            got = proxy.call("flesch_reading_ease", text, catch_errors=True)
            if proxy.last_error is not None:
                if first_error is None:
                    first_error = f"case #{case_index} failed: {proxy.last_error}"
                continue
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                continue
            delta = abs(float(got) - self.expected[case_index])
            errors.append(delta)
            if delta <= TOLERANCE:
                passed += 1

        total = len(CASES)
        pass_rate = passed / total
        mean_abs_error = sum(errors) / len(errors) if errors else 0.0

        if first_error is not None:
            info = (f"{passed}/{total} texts within {TOLERANCE:g} before the agent "
                    f"crashed. {first_error}")
        elif passed == total:
            info = f"All {total} texts match the reference."
        else:
            info = (f"{passed}/{total} texts within {TOLERANCE:g}; "
                    f"mean absolute error {mean_abs_error:.4f}.")

        return {
            "agent_index": index,
            "score": round(pass_rate, 6),
            "steps": total,
            "info_message": info,
            "metrics_detail": {
                "pass_rate": round(pass_rate, 6),
                "mean_abs_error": round(mean_abs_error, 6),
            },
        }
