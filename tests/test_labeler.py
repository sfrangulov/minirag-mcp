import os
import threading
import time
from pathlib import Path

import pytest

from conftest import FakeChoiceModel, FakeModelFactory
from minirag_mcp.labels import JobReport, Labeler, LabelerSource, model, taxonomy_path
from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.model import ModelError
from minirag_mcp.labels.resolve import DocLabels, Label
from minirag_mcp.labels.taxonomy import load_taxonomy

TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.project]
default = "core"
values.core = { description = "The core project." }
values.pilot = { description = "The pilot project.", paths = ["pilot"] }

[facets.type]
question = "What kind of document is this? Judge by form."
values.spec = { description = "Spec. Requirements.", paths = ["specs", "pilot/specs"] }
values.minutes = { description = "Minutes. Date, participants, decisions." }
values.other = { description = "Other. Anything else." }
"""

# Two facets the model answers, one of them a path rule it may override.
TWO_MODEL_FACETS = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.project]
default = "core"
values.core = { description = "The core project." }
values.pilot = { description = "The pilot project.", paths = ["pilot"] }

[facets.type]
override_at = 0.9
values.spec = { description = "Spec.", paths = ["specs"] }
values.minutes = { description = "Minutes." }

[facets.topic]
values.billing = { description = "Billing." }
values.travel = { description = "Travel." }
"""

PATH_ONLY = """
[facets.project]
default = "core"
values.core = { description = "The core project." }
values.pilot = { description = "The pilot project.", paths = ["pilot"] }
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "docs"
    r.mkdir()
    return r


def write_taxonomy(root: Path, text: str = TAXONOMY) -> Path:
    path = taxonomy_path([root])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_labeler(root: Path, factory: FakeModelFactory, text: str = TAXONOMY) -> Labeler:
    return Labeler(
        load_taxonomy(write_taxonomy(root, text)), [root], root.parent / "cache", factory
    )


def label_file(
    labeler: Labeler,
    root: Path,
    rel: str,
    text: str = "Plain body text.",
    file_time: float | None = None,
    trace: dict[str, dict[str, float]] | None = None,
) -> DocLabels:
    return labeler.label(
        source=str(root / rel),
        source_type="file",
        title=Path(rel).stem,
        chunk_texts=[text],
        file_time=file_time,
        trace=trace,
    )


def test_the_model_loads_once_per_job_and_closes_after(root, model_factory):
    model_factory.default = {"minutes": 0.75, "other": 0.25}
    labeler = make_labeler(root, model_factory)

    with labeler.job() as outer:
        with labeler.job() as inner:
            assert inner is outer
            docs = [label_file(labeler, root, f"notes/n{i}.md") for i in range(2)]
        assert not model_factory.created[0].closed
        docs.append(label_file(labeler, root, "notes/n2.md"))
        assert len(model_factory.created) == 1
        assert not model_factory.created[0].closed

    assert model_factory.created[0].closed
    assert len(model_factory.created[0].calls) == 3
    assert model_factory.calls == [(labeler.taxonomy.model, root.parent / "cache")]
    assert outer.problems == []
    for doc in docs:
        assert doc.labels == {
            "project": Label("core", 1.0, "default"),
            "type": Label("minutes", 0.75, "model", second="other"),
        }
        assert doc.taxonomy_hash == labeler.taxonomy.hash


def test_label_outside_a_job_is_its_own_job(root, model_factory):
    labeler = make_labeler(root, model_factory)

    label_file(labeler, root, "notes/a.md")
    label_file(labeler, root, "notes/b.md")

    assert len(model_factory.created) == 2
    assert all(m.closed for m in model_factory.created)


def test_path_and_default_facets_never_load_the_model(root, model_factory):
    labeler = make_labeler(root, model_factory)

    with labeler.job() as report:
        by_default = label_file(labeler, root, "specs/a.pdf")
        by_path = label_file(labeler, root, "pilot/specs/b.pdf")

    assert model_factory.calls == []
    assert report.problems == []
    assert by_default.labels == {
        "project": Label("core", 1.0, "default"),
        "type": Label("spec", 1.0, "path"),
    }
    assert by_path.labels == {
        "project": Label("pilot", 1.0, "path"),
        "type": Label("spec", 1.0, "path"),
    }


def test_a_model_that_cannot_load_is_reported_once_and_not_retried(root, model_factory):
    model_factory.error = ModelError("needs the extra")
    labeler = make_labeler(root, model_factory)

    with labeler.job() as report:
        docs = [label_file(labeler, root, f"pilot/n{i}.md") for i in range(3)]

    assert [d.labels for d in docs] == [{"project": Label("pilot", 1.0, "path")}] * 3
    assert report.problems == ["needs the extra"]
    assert len(model_factory.calls) == 1

    model_factory.error = None
    model_factory.default = {"minutes": 1.0}
    with labeler.job() as report:
        doc = label_file(labeler, root, "pilot/n0.md")
    assert report.problems == []
    assert doc.labels["type"].value == "minutes"


def test_one_failing_document_does_not_poison_the_job(root, model_factory):
    model_factory.fail_on = "HUGE"
    model_factory.default = {"minutes": 0.875, "other": 0.125}
    labeler = make_labeler(root, model_factory)
    trace: dict[str, dict[str, float]] = {}

    with labeler.job() as report:
        bad = label_file(labeler, root, "notes/bad.md", text="A HUGE document.", trace=trace)
        good = label_file(labeler, root, "notes/good.md")

    assert bad.labels == {"project": Label("core", 1.0, "default")}
    assert trace == {}
    assert good.labels["type"] == Label("minutes", 0.875, "model", second="other")
    assert report.problems == [
        f"{root / 'notes' / 'bad.md'}: cannot read a document containing HUGE"
    ]
    assert len(model_factory.created) == 1


def test_a_failing_document_keeps_the_path_value_the_model_could_override(root, model_factory):
    model_factory.fail_on = "HUGE"
    labeler = make_labeler(root, model_factory, TWO_MODEL_FACETS)

    with labeler.job() as report:
        doc = label_file(labeler, root, "specs/a.pdf", text="A HUGE spec.")

    assert doc.labels == {
        "project": Label("core", 1.0, "default"),
        "type": Label("spec", 1.0, "path"),
    }
    assert len(report.problems) == 1
    assert len(model_factory.created[0].calls) == 1


def test_an_unexpected_model_exception_is_reported_not_raised(root, model_factory):
    class Broken(FakeChoiceModel):
        def probabilities(self, evidence, question, options):
            raise RuntimeError("llama_decode returned -3")

    model_factory.model_class = Broken
    labeler = make_labeler(root, model_factory)
    with labeler.job() as report:
        doc = label_file(labeler, root, "notes/a.md")
    assert doc.labels == {"project": Label("core", 1.0, "default")}
    assert report.problems == [f"{root / 'notes' / 'a.md'}: RuntimeError: llama_decode returned -3"]

    model_factory.error = ImportError("No module named 'huggingface_hub'")
    with labeler.job() as report:
        doc = label_file(labeler, root, "notes/a.md")
    assert doc.labels == {"project": Label("core", 1.0, "default")}
    assert report.problems == ["ImportError: No module named 'huggingface_hub'"]


def test_evidence_is_the_file_name_label_and_6000_collapsed_chars(root, model_factory):
    taxonomy = TAXONOMY.replace(
        "temperature = 1.0", 'temperature = 1.0\nfile_name_label = "Имя файла"'
    )
    labeler = make_labeler(root, model_factory, taxonomy)
    words = [f"w{i}" for i in range(2000)]
    collapsed = " ".join(words)
    assert len(collapsed) > 6000

    labeler.label(
        source=str(root / "notes" / "a.pdf"),
        source_type="file",
        title="Ignored title",
        chunk_texts=["  ".join(words[:700]) + "\n\n", "\t" + "\n \t".join(words[700:])],
        file_time=None,
    )

    [(question, evidence)] = model_factory.created[0].calls
    assert evidence == "Имя файла: a.pdf\n" + collapsed[:6000]
    assert "w699 w700" in evidence
    assert question == "What kind of document is this? Judge by form."


def test_text_documents_use_their_title_and_no_path_rules(root, model_factory):
    model_factory.default = {"minutes": 0.625, "other": 0.375}
    labeler = make_labeler(root, model_factory)

    doc = labeler.label(
        source=str(root / "pilot" / "specs" / "2024-03-05 notes.md"),
        source_type="data",
        title="Weekly sync notes",
        chunk_texts=["Body."],
        file_time=None,
    )

    assert doc.labels == {
        "type": Label("minutes", 0.625, "model", second="other", uncertain=True),
    }
    assert doc.date == DocDate("", "none")
    [(_, evidence)] = model_factory.created[0].calls
    assert evidence == "File name: Weekly sync notes\nBody."


def test_dates_do_not_need_the_model(root, model_factory):
    model_factory.error = ModelError("needs the extra")
    labeler = make_labeler(root, model_factory)

    with labeler.job():
        by_name = label_file(
            labeler, root, "notes/2024-03-05 plan.pdf", text="No date here.", file_time=0.0
        )
        by_header = label_file(
            labeler, root, "notes/plan.pdf", text="Status 2024-04-01\n" + "filler words " * 30
        )
        across_chunks = labeler.label(
            source=str(root / "notes" / "plan.pdf"),
            source_type="file",
            title="plan",
            chunk_texts=["Status report", "2024-04-01 review"],
            file_time=None,
        )

    assert by_name.labels == {"project": Label("core", 1.0, "default")}
    assert by_name.date == DocDate("2024-03-05", "name")
    assert by_header.date == DocDate("2024-04-01", "document")
    assert across_chunks.date == DocDate("", "none")


def test_trace_records_every_distribution_asked(root, model_factory):
    model_factory.answers = {
        "invoice": {"spec": 0.25, "minutes": 0.75, "billing": 0.875, "travel": 0.125},
        "agenda": {"spec": 0.0625, "minutes": 0.9375, "billing": 0.5, "travel": 0.5},
    }
    labeler = make_labeler(root, model_factory, TWO_MODEL_FACETS)

    kept: dict[str, dict[str, float]] = {}
    kept_doc = label_file(labeler, root, "specs/a.pdf", text="An invoice summary.", trace=kept)
    overridden: dict[str, dict[str, float]] = {}
    overridden_doc = label_file(
        labeler, root, "specs/b.pdf", text="The meeting agenda.", trace=overridden
    )

    assert kept == {
        "type": {"spec": 0.25, "minutes": 0.75},
        "topic": {"billing": 0.875, "travel": 0.125},
    }
    assert kept_doc.labels["type"] == Label("spec", 1.0, "path")
    assert overridden == {
        "type": {"spec": 0.0625, "minutes": 0.9375},
        "topic": {"billing": 0.5, "travel": 0.5},
    }
    assert overridden_doc.labels["type"] == Label("minutes", 0.9375, "model", second="spec")
    assert overridden_doc.labels["topic"].uncertain


class ExclusiveChoiceModel(FakeChoiceModel):
    """Records every call that enters probabilities() while another is still inside it.

    The sleep holds the call open long enough for an unguarded second thread to walk in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inside = threading.Lock()
        self.overlaps: list[str] = []

    def probabilities(self, evidence, question, options):
        if not self._inside.acquire(blocking=False):
            self.overlaps.append(evidence)
            raise AssertionError("probabilities() entered while another call is inside it")
        try:
            time.sleep(0.002)
            return super().probabilities(evidence, question, options)
        finally:
            self._inside.release()


def test_concurrent_jobs_never_enter_the_model_together(root, model_factory):
    model_factory.model_class = ExclusiveChoiceModel
    model_factory.default = {"minutes": 1.0}
    labeler = make_labeler(root, model_factory)
    both_inside = threading.Barrier(2)
    reports: list[JobReport] = []
    errors: list[BaseException] = []
    labelled: list[DocLabels] = []

    def work(folder: str) -> None:
        try:
            with labeler.job() as report:
                reports.append(report)
                both_inside.wait(timeout=10)
                for i in range(20):
                    labelled.append(label_file(labeler, root, f"{folder}/n{i}.md"))
        except BaseException as e:  # surfaced below; an uncaught one would only warn
            errors.append(e)

    threads = [threading.Thread(target=work, args=(f,)) for f in ("sync", "ingest")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(reports) == 2 and reports[0] is reports[1]
    assert reports[0].problems == []
    assert len(model_factory.created) == 1
    shared = model_factory.created[0]
    assert shared.overlaps == []
    assert len(shared.calls) == 40
    assert shared.closed
    assert [d.labels["type"].value for d in labelled] == ["minutes"] * 40


def test_a_document_without_model_facets_does_not_wait_for_another_threads_call(
    root, model_factory
):
    inside, release = threading.Event(), threading.Event()

    class Slow(FakeChoiceModel):
        def probabilities(self, evidence, question, options):
            inside.set()
            release.wait(timeout=10)
            return super().probabilities(evidence, question, options)

    model_factory.model_class = Slow
    labeler = make_labeler(root, model_factory)
    slow = threading.Thread(target=label_file, args=(labeler, root, "notes/slow.md"))
    by_path: list[DocLabels] = []
    quick = threading.Thread(
        target=lambda: by_path.append(label_file(labeler, root, "specs/a.pdf"))
    )
    slow.start()
    try:
        assert inside.wait(timeout=10)
        quick.start()
        quick.join(timeout=2)
        assert not quick.is_alive()
        assert by_path[0].labels["type"] == Label("spec", 1.0, "path")
    finally:
        release.set()
        slow.join(timeout=10)
        quick.join(timeout=10)
    assert len(model_factory.created) == 1 and model_factory.created[0].closed


def test_model_state_is_loaded_unavailable_or_on_demand(root, model_factory, monkeypatch):
    labeler = make_labeler(root, model_factory)

    monkeypatch.setattr(model, "available", lambda: True)
    assert labeler.model_state() == "on demand"
    monkeypatch.setattr(model, "available", lambda: False)
    assert labeler.model_state() == "unavailable"
    with labeler.job():
        label_file(labeler, root, "notes/a.md")
        assert labeler.model_state() == "loaded"
    assert labeler.model_state() == "unavailable"

    path_only = make_labeler(root, model_factory, PATH_ONLY)
    assert path_only.model_state() == "on demand"


def test_a_report_keeps_the_first_occurrence_of_each_problem():
    report = JobReport()
    for problem in ("b", "a", "b", "a", "c"):
        report.add(problem)
    assert report.problems == ["b", "a", "c"]


def set_mtime(path: Path, seconds: int) -> None:
    os.utime(path, ns=(seconds * 10**9, seconds * 10**9))


def test_source_reloads_an_edited_file(root, model_factory):
    source = LabelerSource([root], root.parent / "cache", model_factory)
    assert source.path == root / ".minirag" / "classify.toml"
    assert source.get() == (None, None)

    path = write_taxonomy(root)
    set_mtime(path, 1_700_000_000)
    first, error = source.get()
    assert error is None and first is not None
    assert source.get() == (first, None)
    assert label_file(first, root, "pilot/a.md").labels["project"].value == "pilot"
    assert model_factory.calls == [(first.taxonomy.model, root.parent / "cache")]

    path.write_text(TAXONOMY + 'values.draft = { description = "Draft." }\n', encoding="utf-8")
    set_mtime(path, 1_700_000_001)
    second, error = source.get()
    assert error is None and second is not first
    assert second.taxonomy.facet("type").value_ids == ("spec", "minutes", "other", "draft")
    assert second.taxonomy.hash != first.taxonomy.hash
    assert source.get()[0] is second

    path.write_text(TAXONOMY + 'colour = "red"\n', encoding="utf-8")
    set_mtime(path, 1_700_000_002)
    labeler, error = source.get()
    assert labeler is None
    assert error is not None and error.endswith("facets.type.colour: unknown key")

    path.unlink()
    assert source.get() == (None, None)


def test_source_notices_a_change_in_either_mtime_or_size(root, model_factory):
    source = LabelerSource([root], root.parent / "cache", model_factory)
    path = write_taxonomy(root)
    set_mtime(path, 1_700_000_000)
    first, _ = source.get()

    same_size = TAXONOMY.replace("temperature = 1.0", "temperature = 2.0")
    assert len(same_size) == len(TAXONOMY)
    path.write_text(same_size, encoding="utf-8")
    set_mtime(path, 1_700_000_001)
    by_mtime, _ = source.get()
    assert by_mtime is not first
    assert by_mtime.taxonomy.model.temperature == 2.0

    path.write_text(same_size + "\n", encoding="utf-8")
    set_mtime(path, 1_700_000_001)
    by_size, _ = source.get()
    assert by_size is not by_mtime
