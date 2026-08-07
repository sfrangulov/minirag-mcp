from minirag_mcp.chunker.semantic import cosine, merge_blocks
from minirag_mcp.chunker.structural import Block


class StubEmbedder:
    """Maps exact text -> vector; unknown text -> orthogonal-ish default."""

    def __init__(self, table):
        self.table = table

    def __call__(self, texts):
        return [self.table[t] for t in texts]


def T(text):
    return Block(text=text, is_code=False)


def C(text):
    return Block(text=text, is_code=True)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 0]) == 0.0  # zero vector guard


def test_empty():
    assert merge_blocks([], StubEmbedder({})) == []


def test_similar_neighbors_merge():
    embed = StubEmbedder({"a": [1.0, 0.0], "b": [0.99, 0.1], "c": [-1.0, 0.0]})
    out = merge_blocks([T("a"), T("b"), T("c")], embed, min_length=1)
    assert out == ["a\n\nb", "c"]


def test_max_chars_stops_merge():
    embed = StubEmbedder({"x" * 60: [1.0, 0.0], "y" * 60: [1.0, 0.0]})
    out = merge_blocks([T("x" * 60), T("y" * 60)], embed, max_chars=100, min_length=1)
    assert len(out) == 2  # merged would be 122 chars > 100


def test_code_block_merges_without_similarity():
    embed = StubEmbedder({"intro text": [1.0, 0.0], "```\ncode\n```": [-1.0, 0.0]})
    out = merge_blocks([T("intro text"), C("```\ncode\n```")], embed, min_length=1)
    assert out == ["intro text\n\n```\ncode\n```"]  # dissimilar but code attaches


def test_short_chunk_folds_into_neighbor():
    embed = StubEmbedder({"tiny": [1.0, 0.0], "long enough paragraph": [-1.0, 0.0]})
    out = merge_blocks([T("tiny"), T("long enough paragraph")], embed, min_length=10)
    assert out == ["tiny\n\nlong enough paragraph"]


def test_dissimilar_stay_separate():
    embed = StubEmbedder({"first topic here": [1.0, 0.0], "second topic here": [0.0, 1.0]})
    out = merge_blocks([T("first topic here"), T("second topic here")], embed, min_length=1)
    assert len(out) == 2
