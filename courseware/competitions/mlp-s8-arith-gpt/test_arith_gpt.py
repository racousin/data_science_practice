#!/usr/bin/env python3
"""Local tests for the Arithmetic GPT challenge — the parser, the weights gate, the scorer.

    cd courseware/competitions
    uv run --with torch --with safetensors --with pytest \\
        pytest mlp-s8-arith-gpt/test_arith_gpt.py -v

Needs `data/` (run prepare_data.py first) for the problem-set and benchmark
tests; the parser and weights-gate tests need nothing.

What is covered, and why:

* the parser, rule by rule — no EOS means no answer, only text before the
  first EOS counts, the answer follows the LAST '#', a leading zero or a sign
  is not an integer, a PAD emitted before EOS spoils the answer;
* every rejection of the weights gate, each through env.py as a
  ParticipantSubmissionError with a message naming the problem (not
  safetensors, oversize, a missing or extra tensor, a stored tied head, a wrong
  shape, an integer dtype, NaN), and fp16/bf16 accepted with the fp32 score;
* the problem sets: the counts, correct answers, 4-digit operands, no
  duplicates and no dev problem in the test set;
* the declared benchmark_expected_score is what env.py returns for
  benchmark.safetensors, and metrics_detail carries exactly the schema keys;
* digit accuracy, rule by rule;
* the teacher's scratchpad formats (local only) fit the token budget and parse
  to the answer;
* decoding budget: a model that never emits EOS on the whole test set is
  scored in far less than DECODE_BUDGET_SEC on this machine.
"""
import csv
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
sys.path.insert(0, str(PKG))

import arith_gpt as ag  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = load("arith_config", PKG / "config.py").CONFIG


class Rejected(Exception):
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """env.py staged next to its private files, as the platform lays out the env dir."""
    if not (DATA / "test_problems.csv").exists():
        pytest.skip("run prepare_data.py first")
    stage = tmp_path_factory.mktemp("env")
    shutil.copy2(PKG / "env.py", stage / "env.py")
    for name in CONFIG["private_files"]:
        src = DATA / name if (DATA / name).exists() else PKG / name
        shutil.copy2(src, stage / name)
    instance = load("arith_env", stage / "env.py").Env(is_evaluation=True)
    instance.ParticipantSubmissionError = Rejected
    return instance


def ids(s: str) -> list[int]:
    return ag.encode(s)


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize("completion, expected", [
    ("#579\n", 579),
    ("1|22>333#7006652\n", 7006652),        # any scratchpad before the #
    ("#0\n", 0),
    ("1#2#34\n", 34),                 # the LAST '#'
    ("#12\n#99\n", 12),               # only text before the first EOS
    ("#12", None),                    # no EOS: out of tokens
    ("579\n", None),                  # no '#'
    ("#\n", None),                    # nothing after '#'
    ("#012\n", None),                 # leading zero
    ("#-12\n", None),                 # sign
    ("#1>2\n", None),                 # not digits
])
def test_parse_answer(completion, expected):
    assert ag.parse_answer(ids(completion)) == expected


def test_pad_before_eos_spoils_the_answer():
    assert ag.parse_answer([ag.STOI["#"], ag.STOI["1"], ag.PAD, ag.STOI["2"], ag.EOS]) is None


# ------------------------------------------------------------------ weights gate
def state(model=None, dtype=torch.float32):
    model = model or ag.GPT()
    return {k: v.to(dtype).contiguous() for k, v in model.state_dict().items() if k != "head.weight"}


@pytest.mark.parametrize("mutate, message", [
    (lambda sd: sd.pop("pos.weight"), "missing ['pos.weight']"),
    (lambda sd: sd.update(extra=torch.zeros(1)), "unexpected ['extra']"),
    (lambda sd: sd.update({"head.weight": sd["tok.weight"].clone()}), "unexpected ['head.weight']"),
    (lambda sd: sd.update({"pos.weight": torch.zeros(128, 256)}), "shape (128, 256)"),
    (lambda sd: sd.update({"ln_f.bias": torch.zeros(ag.CONFIG.n_embd, dtype=torch.int32)}), "torch.int32"),
    (lambda sd: sd["ln_f.weight"].__setitem__(0, float("nan")), "NaN"),
])
def test_bad_weights_are_rejected_with_a_reason(env, tmp_path, mutate, message):
    sd = state()
    mutate(sd)
    path = tmp_path / "weights.safetensors"
    save_file(sd, str(path))
    with pytest.raises(Rejected, match=None) as exc:
        env.evaluate(str(path))
    assert message in str(exc.value)


def test_not_safetensors_is_rejected(env, tmp_path):
    path = tmp_path / "weights.safetensors"
    torch.save(ag.GPT().state_dict(), path)           # a pickle, not safetensors
    with pytest.raises(Rejected, match="not a readable safetensors file"):
        env.evaluate(str(path))


def test_oversize_is_rejected_before_loading(env, tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"\0" * (ag.MAX_WEIGHTS_BYTES + 1))
    with pytest.raises(Rejected, match="the limit is"):
        env.evaluate(str(path))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_is_accepted(tmp_path, dtype):
    torch.manual_seed(0)
    model = ag.GPT()
    path = tmp_path / "w.safetensors"
    ag.save_weights(model, str(path), dtype=dtype)
    loaded = ag.load_weights(str(path))
    assert loaded.head.weight is loaded.tok.weight   # the tie survives loading
    assert torch.allclose(loaded.tok.weight, model.tok.weight.to(dtype).float())


# ------------------------------------------------------------------ problem sets
def read(path):
    with open(path, newline="") as fh:
        return [(int(r["a"]), int(r["b"]), int(r["answer"])) for r in csv.DictReader(fh)]


def test_problem_sets():
    if not (DATA / "test_problems.csv").exists():
        pytest.skip("run prepare_data.py first")
    dev, test = read(DATA / "dev_problems.csv"), read(DATA / "test_problems.csv")
    assert (len(dev), len(test)) == (200, 1000)
    for rows in (dev, test):
        assert len({(a, b) for a, b, _ in rows}) == len(rows)
        for a, b, c in rows:
            assert 1000 <= a <= 9999 and 1000 <= b <= 9999 and c == a * b
    assert not {(a, b) for a, b, _ in dev} & {(a, b) for a, b, _ in test}


# ------------------------------------------------------------------ benchmark
def test_benchmark_score_and_schema(env):
    bench = PKG / CONFIG["benchmark_file"]
    if not bench.exists():
        pytest.skip("run prepare_data.py first")
    row = env.evaluate(str(bench))["agent_results"][0]
    schema = {d["key"] for d in CONFIG["metrics_schema"] if d["source"] == "env"}
    assert set(row["metrics_detail"]) == schema
    assert row["metrics_detail"][CONFIG["metric"]] == row["score"]
    assert abs(row["score"] - CONFIG["benchmark_expected_score"]) <= CONFIG["benchmark_score_tol"]


# ------------------------------------------------------------------ metrics
@pytest.mark.parametrize("got, truth, expected", [
    (7006652, 7006652, 1.0),
    (7006650, 7006652, 6 / 7),        # the units digit wrong
    (706652, 7006652, 5 / 7),         # a digit dropped shifts the rest: aligned from the units
    (None, 7006652, 0.0),
    (17006652, 7006652, 7 / 8),       # an extra digit counts as a wrong one
])
def test_digit_accuracy(got, truth, expected):
    assert ag.digit_accuracy(got, truth) == pytest.approx(expected)


# ------------------------------------------------------------------ format + budget
def test_teacher_formats_fit_the_budget():
    path = PKG / "teacher_reference_solution.py"
    if not path.exists():
        pytest.skip("teacher_reference_solution.py is local only")
    ref = load("arith_ref", path)
    assert len("9999*9999=") + ag.MAX_NEW_TOKENS <= ag.CONFIG.block_size
    for fmt in ref.FORMATS.values():
        assert len(fmt(9999, 9999)) <= ag.MAX_NEW_TOKENS
        for a, b in [(1234, 5678), (1000, 1000), (9999, 9999), (4821, 9170)]:
            assert ag.parse_answer(ids(fmt(a, b))) == a * b


def test_worst_case_decode_is_far_inside_the_budget(env):
    """A model that never ends its answer decodes every problem to the full budget."""
    model = ag.GPT()
    with torch.no_grad():
        # ln_f outputs a constant v, and only the token '1' reads v: it always wins.
        v = torch.ones(ag.CONFIG.n_embd)
        model.ln_f.weight.zero_()
        model.ln_f.bias.copy_(v)
        model.tok.weight.zero_()
        model.tok.weight[ag.STOI["1"]] = v
    torch.set_num_threads(1)
    started = time.monotonic()
    result = ag.evaluate_problems(model, env.problems)
    seconds = time.monotonic() - started
    assert result["answered"] == 0
    # The env container is slower than a laptop.
    budget = sys.modules[type(env).__module__].DECODE_BUDGET_SEC
    assert seconds < budget / 4, seconds


def test_cached_decoding_matches_full_recompute():
    """The KV cache is an optimisation: greedy tokens equal a full forward per step."""
    torch.manual_seed(1)
    model = ag.GPT().eval()
    x = torch.tensor([ids("4821*9170="), ids("1234*5678="), ids("9999*9999=")])
    cached = model.generate(x, max_new_tokens=40)
    seq = x
    with torch.no_grad():
        for _ in range(cached.shape[1]):
            seq = torch.cat([seq, model(seq)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, seq[:, x.shape[1]:])
