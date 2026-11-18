from __future__ import annotations

from score import aggregate, agreement, overall


def _results(*rows):
    return {
        "questions": [
            dict(
                zip(
                    (
                        "question_id",
                        "answer_correct",
                        "completeness_pct",
                        "document_recall_pct",
                        "invalid_extra_docs",
                    ),
                    r,
                    strict=True,
                )
            )
            for r in rows
        ]
    }


def test_overall_is_correct_times_completeness():
    res = _results(("q1", True, 80.0, 100.0, 0), ("q2", False, 100.0, 0.0, 1))
    assert overall(res, ["q1", "q2"]) == 40.0


def test_an_unscored_question_counts_as_zero():
    res = _results(("q1", True, 80.0, 100.0, 0))
    assert overall(res, ["q1", "q2"]) == 40.0
    agg = aggregate(res, ["q1", "q2"], {"q1": ("d",), "q2": ("d",)})
    assert agg["scored"] == 1
    assert agg["correctness"] == 50.0
    assert agg["recall"] == 50.0


def test_recall_only_over_questions_with_gold():
    res = _results(("q1", True, 100.0, 100.0, 0), ("q2", True, 100.0, None, 0))
    assert aggregate(res, ["q1", "q2"], {"q1": ("d",), "q2": ()})["recall"] == 100.0


def test_agreement_counts_matching_verdicts():
    ours = _results(("q1", True, 100.0, 0, 0), ("q2", False, 50.0, 0, 0))
    pub = _results(("q1", True, 90.0, 0, 0), ("q2", True, 50.0, 0, 0))
    a = agreement(ours, pub, ["q1", "q2"])
    assert a["n"] == 2 and a["correct_agree_pct"] == 50.0 and a["completeness_mean_diff"] == 5.0
