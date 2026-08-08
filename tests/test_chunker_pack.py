"""The token budget, table handling, and rebuilding a section from its units."""

import pytest

from conftest import FakeEmbedder
from corpus_samples import (
    SLIDES,
    SPEC,
    SPREADSHEET,
    TRANSCRIPT,
    TRANSCRIPT_TITLE,
    TRANSCRIPT_WITH_SPEAKERS,
)
from minirag_mcp.chunker import DEFAULT_TOKEN_BUDGET, chunk_markdown, join_parent
from minirag_mcp.chunker.pack import join_parent as join
from minirag_mcp.chunker.sections import split_sections
from minirag_mcp.chunker.tokens import estimate_tokens

count = FakeEmbedder().count_tokens
CORPUS = {
    "spec": (SPEC, ""),
    "transcript": (TRANSCRIPT, TRANSCRIPT_TITLE),
    "transcript_speakers": (TRANSCRIPT_WITH_SPEAKERS, "Воркшоп 2"),
    "slides": (SLIDES, ""),
    "spreadsheet": (SPREADSHEET, ""),
}


def parents(chunks):
    grouped: dict[int, list[str]] = {}
    for c in chunks:
        grouped.setdefault(c.parent_index, []).append(c.text)
    return grouped


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize("budget", [DEFAULT_TOKEN_BUDGET, 60, 30])
def test_no_retrieval_unit_exceeds_the_budget(name, budget):
    """The point of the redesign. The encoder does not rank an over-long chunk badly,
    it never sees the tail at all — measured on the live index, 22.8% of every token
    stored was being discarded before it reached the model."""
    markdown, title = CORPUS[name]
    chunks = chunk_markdown(markdown, count_tokens=count, title=title, budget=budget)
    assert chunks
    for c in chunks:
        assert count(c.text) <= budget, repr(c.text)


def test_the_budget_is_counted_in_tokens_not_characters():
    """A character budget over- or under-fills by language: measured over the corpus,
    prose runs at ~3.3 chars/token and table rows at ~2.2. Two documents of equal
    length in characters must therefore chunk differently when they tokenize
    differently."""
    dense = " ".join("а б в г д е ё ж з и" for _ in range(40))  # many short tokens
    sparse = " ".join("длинноесловобезпробелов" for _ in range(35))  # few long ones
    assert abs(len(dense) - len(sparse)) < len(dense) * 0.15
    assert len(chunk_markdown(dense, count_tokens=count)) > len(
        chunk_markdown(sparse, count_tokens=count)
    )


def test_a_word_longer_than_the_budget_is_still_split():
    """Nothing but a code fence may exceed the budget, including a single token."""
    chunks = chunk_markdown("х" * 4000, count_tokens=estimate_tokens, budget=40)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= 40 for c in chunks)
    assert "".join(c.text for c in chunks) == "х" * 4000


def test_code_fences_stay_atomic_even_over_budget():
    """The one deliberate exception: a fence split in the middle is wrong, not merely
    partial, and the property is load-bearing for indexing source files."""
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```"
    chunks = chunk_markdown(f"Вступление.\n\n{code}\n\nПослесловие.", count_tokens=count)
    fenced = [c for c in chunks if "```" in c.text]
    assert len(fenced) == 1
    assert fenced[0].text.count("```") == 2
    assert "line_0 = 0" in fenced[0].text and "line_199 = 199" in fenced[0].text
    assert count(fenced[0].text) > DEFAULT_TOKEN_BUDGET  # over budget on purpose


def test_table_rows_are_never_split_and_the_header_repeats():
    """The three xlsx files in the corpus were being cut mid-row by the old chunker."""
    chunks = chunk_markdown(SPREADSHEET, count_tokens=count, budget=60)
    row_chunks = [c for c in chunks if "| 1.0 |" in c.text or "| 2.0 |" in c.text]
    assert len(row_chunks) >= 2, "the sheet has to have been split to prove anything"
    for c in row_chunks:
        assert "| Unnamed: 0 | Unnamed: 1 | Unnamed: 2 |" in c.text
        for line in c.text.split("\n"):
            if line.startswith("|"):
                assert line.endswith("|"), f"row cut mid-way: {line!r}"


def test_a_header_too_wide_to_repeat_is_still_stored_once():
    """Dropping an unrepeatable header would delete the table's column names from the
    index entirely; it is demoted to an ordinary first row instead."""
    header = "| " + " | ".join(f"Колонка номер {i}" for i in range(30)) + " |"
    delim = "| " + " | ".join("---" for _ in range(30)) + " |"
    rows = "\n".join(f"| значение {i} |" for i in range(6))
    chunks = chunk_markdown(f"{header}\n{delim}\n{rows}", count_tokens=count, budget=60)
    texts = [c.text for c in chunks]
    assert sum(t.count("Колонка номер 0") for t in texts) == 1
    assert all(count(t) <= 60 for t in texts)


def test_transcript_units_carry_the_window_label_and_the_meeting_title():
    chunks = chunk_markdown(TRANSCRIPT, count_tokens=count, title=TRANSCRIPT_TITLE)
    labelled = [c for c in chunks if c.text.startswith("[")]
    assert labelled
    for c in labelled:
        head = c.text.split("\n", 1)[0]
        assert head.startswith("[00:00–02:00] ") or head.startswith("[02:00–04:00] ")
        assert TRANSCRIPT_TITLE in head


def test_a_prefix_too_long_for_the_budget_degrades():
    """The prefix ladder only means anything if the packer walks it. A `_choose_prefix`
    that returned the first candidate unconditionally passed every test this project
    had — the one test that claimed to cover this only inspected the ladder that
    `sections.py` builds, which is the same tuple whatever the packer does with it."""
    title = "Совещание руководителей " * 60
    chunks = chunk_markdown(TRANSCRIPT, count_tokens=count, title=title)
    labelled = [c for c in chunks if c.text.startswith("[")]
    assert labelled, "the transcript windows have to be labelled at all"
    for c in labelled:
        assert c.text.split("\n", 1)[0] in ("[00:00–02:00]", "[02:00–04:00]"), c.text[:80]
        assert "Совещание руководителей Совещание" not in c.text, "the title did not fit"
    assert all(count(c.text) <= DEFAULT_TOKEN_BUDGET for c in chunks)


def test_heading_units_carry_the_breadcrumb():
    chunks = chunk_markdown(SPEC, count_tokens=count, budget=40)
    glossary = [c for c in chunks if "Бизнес-процесс" in c.text]
    assert glossary
    for c in glossary:
        assert c.text.startswith("1 Общие положения > 1.1 Глоссарий\n\n")


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize("budget", [DEFAULT_TOKEN_BUDGET, 40])
def test_parent_reconstruction_returns_the_section(name, budget):
    """The section is never stored twice — it is exactly its chunks, in order."""
    markdown, title = CORPUS[name]
    chunks = chunk_markdown(markdown, count_tokens=count, title=title, budget=budget)
    sections = split_sections(markdown, title=title)
    assert len(parents(chunks)) == len(sections)
    for index, texts in parents(chunks).items():
        rebuilt = join_parent(texts)
        for word in sections[index].body.split():
            assert word in rebuilt, (name, index, word)


def test_parent_reconstruction_takes_the_siblings_in_order():
    body = "\n\n".join(f"Абзац номер {i} про хранение запасов на складах." for i in range(20))
    chunks = chunk_markdown(f"# 1 Раздел\n\n{body}\n\n# 2 Другой\n\nКоротко.", count_tokens=count)
    multi = [t for t in parents(chunks).values() if len(t) > 1]
    assert multi, "need a section split across several units for this to mean anything"
    for texts in multi:
        rebuilt = join_parent(texts)
        positions = [rebuilt.find(t.split("\n")[-1]) for t in texts]
        assert positions == sorted(positions) and -1 not in positions
        # shuffling the siblings must change the result, or order proves nothing
        assert join_parent(list(reversed(texts))) != rebuilt


def test_the_repeated_header_appears_once_in_the_parent():
    chunks = chunk_markdown(SPREADSHEET, count_tokens=count, budget=60)
    for texts in parents(chunks).values():
        rebuilt = join_parent(texts)
        assert rebuilt.count("| Unnamed: 0 | Unnamed: 1 | Unnamed: 2 |") <= 1
        assert rebuilt.count("ДАННЫЕ ПО ПИЛОТУ 2 «ВАЛЮТНЫЙ КОНТРОЛЬ»") == 1
        # a header row planted mid-table would break the markdown table
        assert "| --- | --- | --- |\n| NaN | ДАННЫЕ" in rebuilt


def test_the_repeated_breadcrumb_appears_once_in_the_parent():
    chunks = chunk_markdown(SPEC, count_tokens=count, budget=40)
    for texts in parents(chunks).values():
        breadcrumb = texts[0].split("\n", 1)[0]
        if len(texts) > 1 and all(t.startswith(breadcrumb) for t in texts):
            assert join_parent(texts).count(breadcrumb) == 1, breadcrumb


def test_join_parent_edge_cases():
    assert join([]) == ""
    assert join(["only"]) == "only"
    assert join(["a", "  ", "b"]) == "a\n\nb"
    assert join(["| x |", "| y |"]) == "| x |\n| y |"
    assert join(["head\n\nfirst", "head\n\nsecond"]) == "head\n\nfirst\n\nsecond"


def test_the_seeded_title_is_counted_against_the_budget():
    """Seeding a *finished* chunk is how 3 of 64,692 chunks on the real corpus ended up
    over budget — every one of them chunk 0. The title goes in before packing."""
    title = "И-41 БНУ Налоговый учет транспортный, имущественный, земельный налоги, эмиссии"
    markdown = "**Функциональная спецификация**\n\n" + "Требование к системе. " * 60
    chunks = chunk_markdown(markdown, count_tokens=count, title=title)
    assert chunks[0].text.startswith(f"# {title}\n\n")
    assert all(count(c.text) <= DEFAULT_TOKEN_BUDGET for c in chunks)
    assert sum(c.text.count(title) for c in chunks) == 1, "chunk 0 only"


def test_a_childless_bodiless_heading_reaches_a_chunk():
    """Where the data loss actually shows: the section was dropped before packing, so
    `Резервные требования` — a whole numbered clause of the spec — was in no chunk, in
    no breadcrumb, and therefore in nothing the index could return."""
    markdown = (
        "# 1 Общие положения\n\nТекст первого раздела.\n\n"
        "# 2 Резервные требования\n\n"
        "# 3 Прочее\n\nТекст третьего раздела.\n"
    )
    chunks = chunk_markdown(markdown, count_tokens=count)
    assert any("2 Резервные требования" in c.text for c in chunks), [c.text for c in chunks]
    assert all(count(c.text) <= DEFAULT_TOKEN_BUDGET for c in chunks)


def test_a_document_that_already_carries_its_title_is_not_seeded():
    chunks = chunk_markdown("# MD Title\n\nBody here.", count_tokens=count, title="MD Title")
    assert chunks[0].text == "# MD Title\n\nBody here."
