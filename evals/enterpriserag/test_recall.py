from __future__ import annotations

from recall import gate, ranked_docs, recall_at, reciprocal_rank, summarize

A = "dsid_" + "a" * 32
B = "dsid_" + "b" * 32
C = "dsid_" + "c" * 32


def _src(name: str) -> dict:
    return {"source": f"/root/slack/x/{name}__m.txt", "displayPath": f"slack/x/{name}__m.txt"}


def test_ranked_docs_are_distinct_ids_in_rank_order():
    payload = {"sources": [_src(B), _src(A), {"source": "/root/readme.txt"}, _src(B)]}
    assert ranked_docs(payload) == [B, A]


def test_recall_at_counts_gold_within_k():
    assert recall_at([A, B], (B, C), 1) == 0.0
    assert recall_at([A, B], (B, C), 2) == 50.0
    assert recall_at([B, C], (B, C), 2) == 100.0


def test_reciprocal_rank_of_the_first_gold():
    assert reciprocal_rank([A, B], (B,)) == 0.5
    assert reciprocal_rank([A], (B,)) == 0.0


def _row(qtype: str, ranked: list[str], gold: tuple[str, ...]) -> dict:
    return {"question_type": qtype, "gold": list(gold), "ranked": ranked, "default": ranked[:2]}


def test_summarize_overall_and_by_type():
    rows = [_row("basic", [A], (A,)), _row("basic", [B], (A,)), _row("semantic", [C, A], (A,))]
    s = summarize(rows)
    assert s["overall"]["n"] == 3
    assert s["overall"]["recall@10"] == round(200 / 3, 2)
    assert s["by_type"]["basic"]["recall@10"] == 50.0
    assert s["by_type"]["semantic"]["mrr"] == 0.5


def _summary(r10: float, mrr: float, per_type: float = 50.0) -> dict:
    return {
        "overall": {"recall@10": r10, "mrr": mrr},
        "by_type": {"basic": {"recall@10": per_type, "mrr": mrr}},
    }


def test_gate_fires_on_an_overall_recall_drop_over_one_point():
    assert gate(_summary(78.9, 0.6), _summary(80.0, 0.6))
    assert not gate(_summary(79.1, 0.6), _summary(80.0, 0.6))


def test_gate_fires_on_an_mrr_drop_over_two_hundredths():
    assert gate(_summary(80.0, 0.57), _summary(80.0, 0.6))
    assert not gate(_summary(80.0, 0.59), _summary(80.0, 0.6))


def test_gate_ignores_a_per_type_drop_alone():
    assert not gate(_summary(80.0, 0.6, per_type=40.0), _summary(80.0, 0.6, per_type=50.0))


def test_deep_recall_bounds_what_reordering_can_win():
    filler = [f"dsid_{i:032x}" for i in range(1, 60)]
    row = {
        "question_type": "basic",
        "gold": [A],
        "ranked": filler[:20],
        "default": filler[:3],
        "pool": 40,
        "pool_recall": 100.0,
        "deep": filler[:29] + [A] + filler[29:],
    }
    m = summarize([row])["overall"]
    assert m["recall@10"] == 0.0
    assert m["deep_recall@20"] == 0.0 and m["deep_recall@50"] == 100.0
    assert m["recall_pool"] == 100.0 and m["pool_docs"] == 40
