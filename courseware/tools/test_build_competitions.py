#!/usr/bin/env python3
"""Tests for build_competitions.py's per-package settings.

    uv run --with pytest pytest courseware/tools/test_build_competitions.py -v

No network and no SDK: a fake client records every call `build_one` and
`refresh_one` make, against packages written into a temporary directory. What
is covered, and why each one exists:

* a package with none of the optional keys makes exactly the calls it made
  before they existed — `submission.csv` through the text route, no engine
  call, no extra settings — because six live challenges are built that way;
* a `submission.csv.gz` package sends its name and upload cap as settings,
  pins its engine first, and uploads its benchmark as bytes under the
  configured name — the multipart route stores a file under the part's name,
  so uploading `benchmark_submission.csv.gz` without `filename=` would leave
  the benchmark run looking for a file that is not there;
* public files that are hand-written (EXPERTISE.md) are found in the package
  directory, and private labels need not be called `y_test.csv`;
* the server disagreeing with the package (engine, submission name) stops the
  build rather than letting the benchmark fail one step later;
* a missing file stops `refresh` before it stops the live challenge;
* a package can only be selected under its own course, so `publish` on one
  course never reaches another's challenge;
* the course bar: a `pass_threshold` above the benchmark is a typo unless the
  package declares `expert_expected_score` (what its own reference solution
  reaches) at or above it — DPE's 0.85 bar over a 0.7606 benchmark is the
  case — and a None in either key reads as absent, so `attach` falls back to
  the benchmark score.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_competitions as bc  # noqa: E402

SCHEMA = [{"key": "auc", "label": "AUC", "source": "env", "agg": "mean",
           "format": "number", "precision": 4, "higher_is_better": True}]


class FakeClient:
    def __init__(self, *, existing=None, engine_echo=None, stored_name=None):
        self.calls = []
        self.existing = existing or []
        self.engine_echo = engine_echo
        self.stored_name = stored_name
        self.settings = {"submission_filename": "submission.csv",
                         "max_upload_size_bytes": 100 * 1024 * 1024}
        self.benchmark_bytes = {}

    def _log(self, _method, *args, **kwargs):
        self.calls.append((_method, args, kwargs))

    def names(self):
        return [c[0] for c in self.calls]

    def kwargs_of(self, name):
        return [c[2] for c in self.calls if c[0] == name]

    def creator_competitions(self):
        self._log("creator_competitions")
        return self.existing

    def create_competition(self, **kwargs):
        self._log("create_competition", **kwargs)
        return {"competition_id": 7}

    def update_challenge_configuration(self, cid, **fields):
        self._log("update_challenge_configuration", cid, **fields)
        echoed = self.engine_echo if self.engine_echo is not None else fields["engine_id"]
        return {"engine_id": echoed}

    def set_competition_markdown(self, cid, body):
        self._log("set_competition_markdown", cid)

    def update_settings(self, cid, **settings):
        self._log("update_settings", cid, **settings)
        for key in ("submission_filename", "max_upload_size_bytes"):
            if key in settings:
                self.settings[key] = settings[key]
        stored = dict(self.settings)
        if self.stored_name is not None:
            stored["submission_filename"] = self.stored_name
        return stored

    def update_env_file_content(self, cid, name, content):
        self._log("update_env_file_content", cid, name)
        return {"status": "ok"}

    def upload_env_file(self, cid, path):
        self._log("upload_env_file", cid, os.path.basename(path))

    def creator_datasets(self, cid):
        self._log("creator_datasets", cid)
        return {"datasets": []}

    def create_dataset(self, cid, **kwargs):
        self._log("create_dataset", cid, **kwargs)
        return {"id": 3}

    def upload_dataset_file(self, cid, ds_id, path):
        self._log("upload_dataset_file", cid, ds_id, os.path.basename(path))

    def update_benchmark_file_content(self, cid, name, content):
        self._log("update_benchmark_file_content", cid, name)
        self.benchmark_bytes[name] = content.encode()

    def upload_benchmark_file(self, cid, path, filename=None):
        # The SDK names the multipart part `filename`, else the local base name.
        name = filename if filename is not None else os.path.basename(path)
        self._log("upload_benchmark_file", cid, name)
        with open(path, "rb") as fh:
            self.benchmark_bytes[name] = fh.read()
        return {"filename": name}

    def run_benchmark(self, cid):
        self._log("run_benchmark", cid)

    def benchmark_status(self, cid):
        return {"status": "completed", "success": True,
                "agent_results": [{"score": 0.75}]}

    def start_competition(self, cid):
        self._log("start_competition", cid)

    def stop_competition(self, cid):
        self._log("stop_competition", cid)


@pytest.fixture
def packages(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "PACKAGES_DIR", tmp_path)
    monkeypatch.setattr(bc, "STATE_FILE", tmp_path / ".mlarena-state.json")
    return tmp_path


def write_package(root, pkg, config: dict, data: dict, top: dict | None = None):
    d = root / pkg
    (d / "data").mkdir(parents=True)
    (d / "config.py").write_text(f"CONFIG = {config!r}\n")
    (d / "env.py").write_text("class Env: pass\n")
    (d / "overview.md").write_text("# overview\n")
    for name, content in data.items():
        (d / "data" / name).write_bytes(content)
    for name, content in (top or {}).items():
        (d / name).write_bytes(content)
    return d


def base_config(**overrides):
    cfg = {
        "name": "Test challenge", "kernel_version": "file_v1",
        "module_slug": "s2-x", "label": "Test", "metric": "auc",
        "benchmark_expected_score": 0.75, "benchmark_score_tol": 1e-6,
        "dataset_label": "Test data", "dataset_description": "d",
        "deployment_nb_constraint_run": 1, "deployment_nb_initial_score_run": 1,
        "metrics_schema": SCHEMA, "is_public_initial": False,
    }
    cfg.update(overrides)
    return cfg


def legacy_package(root):
    return write_package(
        root, "legacy",
        base_config(public_files=["X_train.csv", "X_test.csv"],
                    private_files=["y_test.csv"],
                    benchmark_file="data/benchmark_submission.csv"),
        {"X_train.csv": b"id,a\n1,2\n", "X_test.csv": b"id,a\n3,4\n",
         "y_test.csv": b"id,prediction\n3,1\n",
         "benchmark_submission.csv": b"id,prediction\n3,0\n"})


def dpe_package(root, bench_bytes=None, **overrides):
    bench = bench_bytes if bench_bytes is not None else gzip.compress(b"id,f1\nte_1,0.5\n")
    cfg = base_config(
        submission_filename="submission.csv.gz",
        max_upload_size_bytes=200 * 1024 * 1024,
        engine_id=42,
        public_files=["train.csv.gz", "test.csv.gz", "EXPERTISE.md",
                      "DICTIONNAIRE.md", "sample_submission.csv.gz"],
        private_files=["labels_train.csv", "labels_test.csv"],
        benchmark_file="benchmark_submission.csv.gz",
    )
    cfg.update(overrides)
    return write_package(
        root, "dpe", cfg,
        {"train.csv.gz": gzip.compress(b"id,x,classe_efg\n"),
         "test.csv.gz": gzip.compress(b"id,x\n"),
         "sample_submission.csv.gz": bench,
         "labels_train.csv": b"id,classe_efg\n", "labels_test.csv": b"id,classe_efg\n"},
        top={"EXPERTISE.md": b"# expertise\n", "DICTIONNAIRE.md": b"# dictionnaire\n",
             "benchmark_submission.csv.gz": bench})


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def test_legacy_package_builds_as_before(packages):
    legacy_package(packages)
    client = FakeClient()
    bc.build_one(client, bc.load_config("legacy"), {}, "http://test")

    assert "update_challenge_configuration" not in client.names()
    assert "upload_benchmark_file" not in client.names()
    (settings,) = client.kwargs_of("update_settings")
    assert "submission_filename" not in settings
    assert "max_upload_size_bytes" not in settings
    assert [c[1] for c in client.calls if c[0] == "update_benchmark_file_content"] == \
        [(7, "submission.csv")]
    assert client.names()[-1] == "start_competition"


def test_gz_package_pins_engine_sets_name_and_uploads_bytes(packages):
    pkg_dir = dpe_package(packages)
    client = FakeClient()
    bc.build_one(client, bc.load_config("dpe"), {}, "http://test")

    names = client.names()
    # The engine is pinned before anything is uploaded to the challenge.
    assert names.index("update_challenge_configuration") < names.index("set_competition_markdown")
    assert client.kwargs_of("update_challenge_configuration") == [{"engine_id": 42}]

    (settings,) = client.kwargs_of("update_settings")
    assert settings["submission_filename"] == "submission.csv.gz"
    assert settings["max_upload_size_bytes"] == 200 * 1024 * 1024

    assert "update_benchmark_file_content" not in names
    assert client.benchmark_bytes == {
        "submission.csv.gz": (pkg_dir / "benchmark_submission.csv.gz").read_bytes()}
    assert names.index("update_settings") < names.index("upload_benchmark_file") \
        < names.index("run_benchmark")

    uploaded = [c[1][2] for c in client.calls if c[0] == "upload_dataset_file"]
    assert uploaded == ["train.csv.gz", "test.csv.gz", "EXPERTISE.md",
                        "DICTIONNAIRE.md", "sample_submission.csv.gz"]
    env_files = [c[1][1] for c in client.calls if c[0] == "upload_env_file"]
    assert env_files == ["labels_train.csv", "labels_test.csv"]


def test_engine_id_none_keeps_the_kind_default(packages):
    dpe_package(packages, engine_id=None)
    client = FakeClient()
    bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert "update_challenge_configuration" not in client.names()


def test_server_reporting_another_engine_stops_the_build(packages):
    dpe_package(packages)
    client = FakeClient(engine_echo=5)
    with pytest.raises(RuntimeError, match="engine 42"):
        bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert "run_benchmark" not in client.names()


def test_server_ignoring_the_submission_name_stops_the_build(packages):
    dpe_package(packages)
    client = FakeClient(stored_name="submission.csv")
    with pytest.raises(RuntimeError, match="submission.csv.gz"):
        bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert "run_benchmark" not in client.names()


def test_uncompressed_gz_benchmark_fails_before_any_request(packages):
    dpe_package(packages, bench_bytes=b"id,f1\nte_1,0.5\n")
    client = FakeClient()
    with pytest.raises(SystemExit, match="not gzip-compressed"):
        bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert client.names() == ["creator_competitions"]


def test_unpinned_benchmark_score_fails_before_create(packages):
    """verify_benchmark skips a None score and the build then starts the
    challenge, so an unpinned package must stop before anything is created."""
    dpe_package(packages, benchmark_expected_score=None)
    client = FakeClient()
    with pytest.raises(SystemExit, match="benchmark_expected_score is not pinned"):
        bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert client.names() == ["creator_competitions"]
    with pytest.raises(SystemExit, match="no bar to attach"):
        bc.pass_threshold(bc.load_config("dpe"))


def test_missing_public_file_fails_before_create(packages):
    pkg_dir = dpe_package(packages)
    (pkg_dir / "EXPERTISE.md").unlink()
    client = FakeClient()
    with pytest.raises(SystemExit, match="EXPERTISE.md"):
        bc.build_one(client, bc.load_config("dpe"), {}, "http://test")
    assert "create_competition" not in client.names()


def test_file_in_data_and_package_dir_with_different_bytes_is_refused(packages):
    pkg_dir = dpe_package(packages)
    (pkg_dir / "data" / "EXPERTISE.md").write_bytes(b"# stale\n")
    with pytest.raises(SystemExit, match="different contents"):
        bc.preflight(bc.load_config("dpe"))


def test_refresh_checks_files_before_stopping_the_challenge(packages):
    pkg_dir = dpe_package(packages)
    (pkg_dir / "data" / "labels_test.csv").unlink()
    client = FakeClient(existing=[{"id": 7, "name": "Test challenge", "is_started": True}])
    with pytest.raises(SystemExit, match="labels_test.csv"):
        bc.refresh_one(client, None, bc.load_config("dpe"), "http://test", keep_agents=True)
    assert "stop_competition" not in client.names()


# --------------------------------------------------------------------------- #
# config validation and package selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overrides,message", [
    ({"submission_filename": "data/submission.csv.gz"}, "bare file name"),
    ({"submission_filename": ""}, "bare file name"),
    # The backend refuses these too (secure_filename would rename them); the
    # build must stop before it creates the challenge, not at update_settings.
    ({"submission_filename": "my submission.csv.gz"}, "bare file name"),
    ({"submission_filename": ".submission.csv.gz"}, "bare file name"),
    ({"submission_filename": "sub..csv.gz"}, "bare file name"),
    ({"submission_filename": "submission.csv.gz_"}, "bare file name"),
    ({"submission_filename": "soumission-é.csv.gz"}, "bare file name"),
    ({"max_upload_size_bytes": "200MB"}, "max_upload_size_bytes"),
    ({"engine_id": True}, "engine_id"),
    ({"engine_id": "42"}, "engine_id"),
    ({"pass_threshold": "0.8"}, "pass_threshold must be a number"),
    ({"pass_threshold": True}, "pass_threshold must be a number"),
    ({"expert_expected_score": "0.9"}, "expert_expected_score must be a number"),
    ({"expert_expected_score": True}, "expert_expected_score must be a number"),
])
def test_malformed_optional_keys_are_refused(packages, overrides, message):
    dpe_package(packages, **overrides)
    with pytest.raises(SystemExit, match=message):
        bc.load_config("dpe")


def test_submission_filename_on_a_flex_package_is_refused(packages):
    dpe_package(packages, kernel_version="flex_v1")
    with pytest.raises(SystemExit, match="file_v1 setting"):
        bc.load_config("dpe")


# --------------------------------------------------------------------------- #
# the course bar (`attach`)
# --------------------------------------------------------------------------- #
def bar_of(packages, **overrides) -> float:
    """The bar `attach` would write for a dpe package with these keys."""
    shutil.rmtree(packages / "dpe", ignore_errors=True)
    dpe_package(packages, **overrides)
    return bc.pass_threshold(bc.load_config("dpe"))


def test_bar_at_or_below_the_benchmark_needs_no_expert_score(packages):
    assert bar_of(packages, pass_threshold=0.7) == 0.7
    assert bar_of(packages, pass_threshold=0.75) == 0.75


def test_bar_above_the_benchmark_is_refused_without_an_expert_score(packages):
    """DPE's case before expert_expected_score existed: 0.85 over 0.7606 had to
    be typed on the link by hand, and `attach` would have reverted it."""
    dpe_package(packages, pass_threshold=0.85)
    with pytest.raises(SystemExit, match=r"pass_threshold 0.85 is above the benchmark's "
                                         r"own score 0.75; declare expert_expected_score"):
        bc.load_config("dpe")


def test_bar_above_the_benchmark_is_accepted_up_to_the_expert_score(packages):
    assert bar_of(packages, pass_threshold=0.85, expert_expected_score=0.92705) == 0.85
    # the bar may equal what the reference solution reaches, not exceed it
    assert bar_of(packages, pass_threshold=0.92705, expert_expected_score=0.92705) == 0.92705


def test_bar_above_the_expert_score_is_refused_naming_both(packages):
    dpe_package(packages, pass_threshold=0.95, expert_expected_score=0.92705)
    with pytest.raises(SystemExit, match=r"pass_threshold 0.95 is above expert_expected_score "
                                         r"0.92705"):
        bc.load_config("dpe")


def test_none_bar_and_expert_score_read_as_absent(packages):
    """A package pins both after its ladder is measured; until then the keys
    are present as None and `attach` must fall back to the benchmark."""
    dpe_package(packages, pass_threshold=None, expert_expected_score=None)
    cfg = bc.load_config("dpe")
    assert "pass_threshold" in cfg and cfg["pass_threshold"] is None
    assert bc.pass_threshold(cfg) == 0.75
    # ...and an unpinned benchmark still has no bar at all
    with pytest.raises(SystemExit, match="no bar to attach"):
        bar_of(packages, pass_threshold=None, expert_expected_score=None,
               benchmark_expected_score=None)


def test_attach_writes_the_declared_bar_not_the_benchmark(packages, monkeypatch, tmp_path):
    """The reconciliation in do_attach compares the live link with
    pass_threshold(); a declared bar above the benchmark must be what it
    writes, not what it reverts."""
    dpe_package(packages, pass_threshold=0.85, expert_expected_score=0.92705)
    bc.write_state("http://test", {"competitions": {
        "dpe": {"id": 191, "module_slug": "s2-x", "label": "Test"}}})
    course_dir = tmp_path / "content" / "c"
    course_dir.mkdir(parents=True)
    (course_dir / ".mlarena-state.json").write_text('{"http://test": {"modules": {"s2-x": 20}}}')
    monkeypatch.setattr(bc, "COURSEWARE", tmp_path)

    written = []

    class Teacher:
        def get_module(self, module_id):
            return {"competitions": [{"competition_id": 191, "pass_threshold": 0.75}]}

        def update_challenge_link(self, module_id, cid, pass_threshold):
            written.append((module_id, cid, pass_threshold))

    monkeypatch.setattr(bc, "connect", lambda scope, base_url: Teacher())

    class Args:
        base_url = "http://test"
        course = "c"
        packages = ["dpe"]

    bc.do_attach(Args)
    assert written == [(20, 191, 0.85)]


def test_attach_skips_a_package_that_declares_no_module(packages, monkeypatch, tmp_path):
    """`"module_slug": None` (s2-dpe-energy-label since 2026-09-15) means the
    challenge is live but attached nowhere: attach must neither look the
    module up nor touch any link, so a hand detach is not undone."""
    dpe_package(packages, pass_threshold=0.85, expert_expected_score=0.92705)
    bc.write_state("http://test", {"competitions": {
        "dpe": {"id": 191, "module_slug": None, "label": "Test"}}})
    course_dir = tmp_path / "content" / "c"
    course_dir.mkdir(parents=True)
    (course_dir / ".mlarena-state.json").write_text('{"http://test": {"modules": {"s2-x": 20}}}')
    monkeypatch.setattr(bc, "COURSEWARE", tmp_path)

    class Teacher:
        def get_module(self, module_id):
            raise AssertionError("a package without a module must not be looked up")

        def attach_competition(self, *a, **k):
            raise AssertionError("must not attach")

        def update_challenge_link(self, *a, **k):
            raise AssertionError("must not touch the link")

    monkeypatch.setattr(bc, "connect", lambda scope, base_url: Teacher())

    class Args:
        base_url = "http://test"
        course = "c"
        packages = ["dpe"]

    bc.do_attach(Args)


def test_packages_are_selected_per_course():
    assert bc.select_packages("python-ai-engineering", None) == \
        bc.PACKAGES_BY_COURSE["python-ai-engineering"]
    assert "s2-dpe-energy-label" not in bc.select_packages("python-ai-engineering", None)
    assert bc.select_packages("ms2a-machine-learning-practice", ["s2-dpe-energy-label"]) == \
        ["s2-dpe-energy-label"]
    with pytest.raises(SystemExit, match="pass --course"):
        bc.select_packages("python-ai-engineering", ["s2-dpe-energy-label"])
    with pytest.raises(SystemExit, match="unknown package"):
        bc.select_packages("python-ai-engineering", ["nope"])


def test_fake_client_matches_the_sdk():
    """The fake above is only evidence if the SDK takes the same calls: every
    method the build makes exists, and accepts the keyword arguments used."""
    pytest.importorskip("requests")
    import inspect
    sdk = os.environ.get("MLARENA_SDK_PATH", str(bc.DEFAULT_SDK))
    if not os.path.isdir(os.path.join(sdk, "mlarena")):
        pytest.skip(f"no SDK checkout at {sdk}")
    sys.path.insert(0, sdk)
    from mlarena.client import MLArenaClient

    for method in [m for m in vars(FakeClient) if not m.startswith("_")
                   and m not in ("names", "kwargs_of")]:
        assert hasattr(MLArenaClient, method), f"the SDK has no {method}()"

    def params(method):
        # Methods are wrapped by functools.wraps; signature() follows __wrapped__.
        return inspect.signature(getattr(MLArenaClient, method)).parameters

    assert {"submission_filename", "max_upload_size_bytes"} <= set(params("update_settings"))
    assert "filename" in params("upload_benchmark_file")
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params("update_challenge_configuration").values())
    assert "engine_id" in MLArenaClient._ADMIN_CONFIGURATION_FIELDS


def test_publish_only_touches_the_selected_course(packages, monkeypatch):
    bc.write_state("http://test", {"competitions": {
        "s2-bike-demand": {"id": 183}, "s2-dpe-energy-label": {"id": 190}}})

    class Args:
        base_url = "http://test"
        packages = bc.select_packages("python-ai-engineering", None)

    published = []

    class Publisher:
        def update_competition(self, cid, is_public):
            published.append((cid, is_public))

    monkeypatch.setattr(bc, "connect", lambda scope, base_url: Publisher())
    bc.do_publish(Args)
    assert published == [(183, True)]
