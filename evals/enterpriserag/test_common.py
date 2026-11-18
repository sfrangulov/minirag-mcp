"""uv run pytest evals/enterpriserag -q"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from common import Question, dsid_of, load_questions, outside_root, pilot_ids
from subset import materialize, subset, subset_hash

A = "dsid_" + "a" * 32
B = "dsid_" + "b" * 32
C = "dsid_" + "c" * 32


def test_dsid_of_reads_the_file_name_prefix():
    assert dsid_of(f"slack/eng/{A}__standup.txt") == A
    assert dsid_of(f"/abs/root/gmail/{B}__thread.txt") == B


def test_dsid_of_ignores_names_without_the_prefix():
    assert dsid_of("slack/eng/standup.txt") is None
    assert dsid_of(f"slack/{A}/standup.txt") is None
    assert dsid_of(f"slack/note-{A}.txt") is None


def _write_questions(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_questions_keeps_gold_ids(tmp_path):
    qfile = tmp_path / "q.jsonl"
    _write_questions(
        qfile,
        [
            {
                "question_id": "qst_0001",
                "question_type": "basic",
                "question": "q?",
                "expected_doc_ids": [A, B],
                "gold_answer": "x",
                "answer_facts": [],
            },
            {
                "question_id": "qst_0002",
                "question_type": "info_not_found",
                "question": "r?",
                "expected_doc_ids": [],
                "gold_answer": "y",
                "answer_facts": [],
            },
        ],
    )
    qs = load_questions(qfile)
    assert qs[0] == Question("qst_0001", "basic", "q?", (A, B))
    assert qs[1].expected_doc_ids == ()


def _questions(types: dict[str, int]) -> list[Question]:
    out, n = [], 0
    for t, count in types.items():
        for _ in range(count):
            n += 1
            out.append(Question(f"qst_{n:04d}", t, "?", ()))
    return out


def test_pilot_ids_takes_two_per_type_deterministically():
    qs = _questions({"basic": 175, "semantic": 125, "high_level": 10, "misc": 1})
    ids = pilot_ids(qs)
    assert ids == pilot_ids(qs)
    by_type = {q.question_id: q.question_type for q in qs}
    counts = {
        t: sum(1 for i in ids if by_type[i] == t) for t in ("basic", "semantic", "high_level")
    }
    assert counts == {"basic": 2, "semantic": 2, "high_level": 2}
    assert sum(1 for i in ids if by_type[i] == "misc") == 1
    assert len(set(ids)) == len(ids)


def test_outside_root_refuses_an_index_inside_the_root(tmp_path):
    (tmp_path / "root" / "idx").mkdir(parents=True)
    with pytest.raises(SystemExit):
        outside_root(tmp_path / "root" / "idx", tmp_path / "root")
    (tmp_path / "elsewhere").mkdir()
    assert (
        outside_root(tmp_path / "elsewhere", tmp_path / "root")
        == (tmp_path / "elsewhere").resolve()
    )


def _files() -> list[str]:
    out = [f"slack/c{i % 7}/dsid_{i:032x}__m.txt" for i in range(800)]
    out += [f"gmail/t/dsid_{10_000 + i:032x}__e.txt" for i in range(200)]
    return sorted(out)


def test_subset_keeps_every_gold_file_and_has_the_requested_size():
    files = _files()
    gold = dsid_of(files[5]), dsid_of(files[900])
    qs = [Question("qst_0001", "basic", "?", gold)]
    picked = subset(files, qs, total=100, seed=1)
    assert len(picked) == 100
    assert files[5] in picked and files[900] in picked
    assert picked == sorted(picked)


def test_subset_is_stable_for_a_seed_and_moves_with_another():
    files = _files()
    a = subset(files, [], total=100, seed=1)
    assert a == subset(files, [], total=100, seed=1)
    assert a != subset(files, [], total=100, seed=2)
    assert subset_hash(a) != subset_hash(subset(files, [], total=100, seed=2))


def test_subset_samples_each_source_in_proportion():
    picked = subset(_files(), [], total=100, seed=3)
    slack = sum(1 for p in picked if p.startswith("slack/"))
    assert 75 <= slack <= 85


def test_materialize_keeps_the_tree(tmp_path):
    src = tmp_path / "corpus"
    (src / "slack" / "c").mkdir(parents=True)
    (src / "slack" / "c" / f"{A}__x.txt").write_text("hello", encoding="utf-8")
    dest = tmp_path / "sub"
    materialize(src, [f"slack/c/{A}__x.txt"], dest)
    assert (dest / "slack" / "c" / f"{A}__x.txt").read_text(encoding="utf-8") == "hello"
