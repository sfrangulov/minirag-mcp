from minirag_mcp.chunker.structural import Block, split_markdown


def test_empty_doc():
    assert split_markdown("") == []
    assert split_markdown("   \n\n  ") == []


def test_paragraphs_are_blocks():
    blocks = split_markdown("Para one.\n\nPara two.")
    assert [b.text for b in blocks] == ["Para one.", "Para two."]
    assert all(not b.is_code for b in blocks)


def test_heading_glues_to_next_paragraph():
    blocks = split_markdown("# Title\n\nIntro text.\n\n## Section\n\nBody.")
    assert blocks[0].text == "# Title\n\nIntro text."
    assert blocks[1].text == "## Section\n\nBody."


def test_trailing_heading_is_own_block():
    blocks = split_markdown("Body.\n\n# Dangling")
    assert blocks[-1].text == "# Dangling"


def test_code_fence_is_atomic_with_blank_lines():
    md = "Intro.\n\n```python\nx = 1\n\n\ny = 2\n```\n\nOutro."
    blocks = split_markdown(md)
    assert [b.is_code for b in blocks] == [False, True, False]
    assert "x = 1\n\n\ny = 2" in blocks[1].text
    assert blocks[1].text.startswith("```python") and blocks[1].text.endswith("```")


def test_tilde_fence():
    blocks = split_markdown("~~~\ncode here\n~~~")
    assert blocks == [Block(text="~~~\ncode here\n~~~", is_code=True)]


def test_unterminated_fence_runs_to_eof():
    blocks = split_markdown("```\nno closing fence\nstill code")
    assert len(blocks) == 1 and blocks[0].is_code


def test_long_code_fence_never_split():
    md = "```\n" + "\n".join(f"line {i}" for i in range(500)) + "\n```"
    blocks = split_markdown(md, max_chars=100)
    assert len(blocks) == 1 and blocks[0].is_code


def test_long_paragraph_split_at_whitespace():
    md = " ".join(f"word{i}" for i in range(400))
    blocks = split_markdown(md, max_chars=200)
    assert len(blocks) > 1
    assert all(len(b.text) <= 200 for b in blocks)
    joined = " ".join(b.text for b in blocks)
    assert joined.split() == md.split()  # no words lost


def test_longer_fence_not_closed_by_shorter_inner_fence():
    md = "````\nouter\n```\ninner\n```\nend\n````"
    blocks = split_markdown(md)
    assert len(blocks) == 1 and blocks[0].is_code


def test_consecutive_headings_glue_together():
    blocks = split_markdown("# A\n## B\n### C\nBody text.")
    assert len(blocks) == 1
    assert blocks[0].text == "# A\n\n## B\n\n### C\n\nBody text."


def test_trailing_consecutive_headings_one_block():
    blocks = split_markdown("Body.\n\n# A\n## B")
    assert blocks[-1].text == "# A\n\n## B"
