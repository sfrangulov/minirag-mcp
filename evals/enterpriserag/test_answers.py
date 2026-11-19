from __future__ import annotations

import json

from answers import build, cited_ids, final_result, tool_calls

A = "dsid_" + "a" * 32
B = "dsid_" + "b" * 32
C = "dsid_" + "c" * 32


def test_the_trailing_block_wins_over_ids_in_the_body():
    answer = f"See {C} for background.\n\nThe limit is 10 MiB.\n\nSources:\n- {A}\n- {B}\n"
    assert cited_ids(answer) == ([A, B], "block")


def test_display_paths_with_directories_yield_their_ids_in_order_once():
    answer = (
        "Answer text.\n\n**Sources**\n"
        f"- github/inference-api/{B}__multipart-limits.txt\n"
        f"- slack/eng-api/{A}__thread.txt\n"
        f"- github/inference-api/{B}__multipart-limits.txt\n"
    )
    assert cited_ids(answer) == ([B, A], "block")


def test_numbered_lists_count_as_a_block():
    answer = f"Answer.\n\nDocuments:\n1. {A}\n2. {B}"
    assert cited_ids(answer) == ([A, B], "block")


def test_no_block_falls_back_to_the_whole_text():
    answer = f"The answer comes from {A}, and {B} agrees."
    assert cited_ids(answer) == ([A, B], "whole")


def test_no_ids_at_all():
    assert cited_ids("I could not find this.") == ([], "none")


def _transcript(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_final_result_is_the_last_result_event(tmp_path):
    t = tmp_path / "q.jsonl"
    _transcript(
        t,
        [
            {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "mcp__minirag__query_documents", "input": {}},
                        {"type": "tool_use", "name": "mcp__minirag__read_file", "input": {}},
                    ]
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "ok",
                "duration_ms": 1000,
                "total_cost_usd": 0.5,
                "num_turns": 3,
            },
        ],
    )
    assert final_result(t)["result"] == "ok"
    assert tool_calls(t) == {"mcp__minirag__query_documents": 1, "mcp__minirag__read_file": 1}


def test_build_answers_every_id_and_counts_what_went_wrong(tmp_path):
    raw = tmp_path / "minirag" / "raw"
    raw.mkdir(parents=True)
    _transcript(
        raw / "qst_0001.jsonl",
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": f"Answer.\n\nSources:\n- {A}",
                "duration_ms": 2000,
                "total_cost_usd": 1.0,
            }
        ],
    )
    _transcript(raw / "qst_0002.jsonl", [{"type": "system", "subtype": "init"}])
    (raw / "qst_0002.timeout").write_text("")
    _transcript(raw / "qst_0003.jsonl", [{"type": "system", "subtype": "init"}])
    _transcript(
        raw / "qst_0004.jsonl",
        [
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "",
                "duration_ms": 10,
                "total_cost_usd": 0.0,
            }
        ],
    )
    stats = build(tmp_path, "minirag", ["qst_0001", "qst_0002", "qst_0003", "qst_0004", "qst_0005"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "minirag" / "answers.jsonl").read_text().splitlines()
    ]
    assert [r["question_id"] for r in rows] == [f"qst_000{i}" for i in range(1, 6)]
    assert rows[0] == {
        "question_id": "qst_0001",
        "answer": f"Answer.\n\nSources:\n- {A}",
        "document_ids": [A],
    }
    assert all(r["answer"] == "" and r["document_ids"] == [] for r in rows[1:])
    assert stats["block"] == 1
    assert stats["timeout"] == 1
    assert stats["error"] == 2
    assert stats["missing"] == 1
