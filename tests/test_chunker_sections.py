"""Category detection and parent sectioning, against the real corpus shapes."""

import pytest

from corpus_samples import (
    SLIDES,
    SPEC,
    SPREADSHEET,
    TRANSCRIPT,
    TRANSCRIPT_TITLE,
    TRANSCRIPT_WITH_SPEAKERS,
)
from minirag_mcp.chunker import sections as sec
from minirag_mcp.chunker.sections import (
    GENERIC,
    HEADINGS,
    clean_heading,
    detect_category,
    format_timestamp,
    split_sections,
)

SLIDES_CATEGORY = sec.SLIDES
TRANSCRIPT_CATEGORY = sec.TRANSCRIPT


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        (TRANSCRIPT, TRANSCRIPT_CATEGORY),
        (TRANSCRIPT_WITH_SPEAKERS, TRANSCRIPT_CATEGORY),
        (SLIDES, SLIDES_CATEGORY),
        (SPEC, HEADINGS),
        (SPREADSHEET, HEADINGS),
    ],
)
def test_category_is_read_off_the_markdown(markdown, expected):
    """Not off the extension: all five of these arrive as `.docx`, `.pptx` or `.xlsx`
    and markitdown flattens them into Markdown before the chunker ever sees them."""
    assert detect_category(markdown) == expected


@pytest.mark.parametrize(
    "markdown",
    [
        "",
        "   \n\n  ",
        "Just one paragraph of prose with no structure at all.",
        "# Only one heading\n\nand a body.",
        "Встреча началась в 12:30 и закончилась в 14:00.",  # times, not a transcript
        "| a | b |\n| --- | --- |\n| 1 | 2 |",
    ],
)
def test_detection_falls_back_to_generic_when_unsure(markdown):
    """Being wrong has to be cheap: an unrecognised document is chunked the old way,
    under the token budget, rather than forced into a category it does not fit."""
    assert detect_category(markdown) == GENERIC


def test_a_time_of_day_in_prose_is_not_a_transcript():
    """The pattern is a timestamp *line*, not a timestamp. Measured on the corpus, the
    107 real transcripts have 50%-52% of their non-blank lines matching and all 452
    other documents have 0% — but a document that lists meeting times could sit in
    between, so the count floor and the share floor both have to hold."""
    minutes = "\n".join(f"Пункт {i} — 1{i}:30 обсуждение." for i in range(10))
    assert detect_category(minutes) == GENERIC


def test_transcript_windows_are_two_minutes_wide():
    sections = split_sections(TRANSCRIPT, title=TRANSCRIPT_TITLE)
    labelled = [s for s in sections if s.prefixes[0].startswith("[")]
    assert [s.prefixes[0] for s in labelled] == [
        f"[00:00–02:00] {TRANSCRIPT_TITLE}",
        f"[02:00–04:00] {TRANSCRIPT_TITLE}",
    ]
    # 0:02 through 1:06 in the first window, 2:05 through 3:40 in the second
    assert "1:06" in labelled[0].body and "2:05" not in labelled[0].body
    assert "2:05" in labelled[1].body and "3:40" in labelled[1].body


def test_transcript_preamble_before_the_first_timestamp_is_kept():
    sections = split_sections(TRANSCRIPT, title=TRANSCRIPT_TITLE)
    assert "010425 Кросс-функциональное демо-ru-RU" in sections[0].body
    assert sections[0].prefixes == ("",)


def test_speaker_prefixed_timestamps_window_the_same_way():
    """Teams keeps the speaker name in front of the timestamp. Both of the corpus's
    transcripts in that format were invisible to a bare-timestamp pattern."""
    sections = split_sections(TRANSCRIPT_WITH_SPEAKERS, title="Воркшоп 2")
    labels = [s.prefixes[0] for s in sections if s.prefixes[0].startswith("[")]
    assert labels == ["[00:00–02:00] Воркшоп 2", "[02:00–04:00] Воркшоп 2"]
    assert "Skachkov, Dmitry" in sections[1].body  # the speaker survives into the text


def test_transcript_prefixes_offer_a_ladder_down_to_the_bare_label():
    """A long meeting title can be most of the budget, so every window offers a shorter
    candidate below the titled one and an empty one below that.

    This pins the ladder `sections.py` builds, and only that — walking it is
    `_choose_prefix`'s job, and it is `test_a_prefix_too_long_for_the_budget_degrades`
    in test_chunker_pack.py that proves the packer actually does. Asserting the ladder
    here and calling it "the label must survive it" is what let a `_choose_prefix` that
    always returned the first candidate pass the whole suite."""
    sections = split_sections(TRANSCRIPT, title="x" * 400)
    window = next(s for s in sections if s.prefixes[0].startswith("["))
    assert window.prefixes[1] == "[00:00–02:00]"
    assert window.prefixes[-1] == ""
    assert [len(p) for p in window.prefixes] == sorted(
        (len(p) for p in window.prefixes), reverse=True
    ), "a ladder is only useful if every rung is shorter than the one above it"


@pytest.mark.parametrize(
    ("seconds", "text"), [(0, "00:00"), (120, "02:00"), (3600, "1:00:00"), (3720, "1:02:00")]
)
def test_timestamp_labels_use_the_transcripts_own_notation(seconds, text):
    assert format_timestamp(seconds) == text


def test_heading_breadcrumb_carries_the_ancestors():
    sections = split_sections(SPEC)
    glossary = next(s for s in sections if "Бизнес-процесс" in s.body)
    assert glossary.prefixes[0] == "1 Общие положения > 1.1 Глоссарий"
    assert glossary.prefixes[1] == "1.1 Глоссарий", "degrades to the innermost heading"


def test_a_heading_with_no_body_survives_in_its_children():
    """`# 1 Общие положения` has no text of its own, only `## 1.1` below it. Emitting
    it as a chunk would add a title-only vector; dropping it loses nothing, because
    every child carries it in the breadcrumb.

    The previous version of this test asserted `not any(s.body.strip() == "")` and
    `all(... for s in sections[:1])` — the first is guaranteed by the empty-body filter
    in `split_sections` and the second by SPEC's ordering, so it passed with the drop
    rule and without it. What has to be pinned is *where* the heading's words went."""
    sections = split_sections(SPEC)
    parent = "1 Общие положения"
    assert not any(s.prefixes[0] == parent for s in sections), "no section of its own"
    assert not any(s.body.strip() == parent for s in sections), "and no chunk of its own"
    children = [s for s in sections if s.prefixes[0].startswith(f"{parent} > ")]
    assert children, "so a child's breadcrumb has to be carrying it"
    assert [s.prefixes[0] for s in children] == [f"{parent} > 1.1 Глоссарий"]


def test_a_childless_heading_with_no_body_becomes_its_own_section():
    """A heading is content — often the most searchable line in a specification. When
    no child carries it into a breadcrumb, its words used to reach neither a chunk nor
    a stored prefix: the section was kept with an empty body and then filtered out by
    `split_sections`, so the heading was absent from the index entirely."""
    markdown = (
        "# 1 Общие положения\n\nТекст первого раздела.\n\n"
        "# 2 Резервные требования\n\n"
        "# 3 Прочее\n\nТекст третьего раздела.\n"
    )
    sections = split_sections(markdown)
    bodies = [s.body for s in sections]
    assert "2 Резервные требования" in bodies, bodies
    assert all(s.body.strip() for s in sections)
    # its own text is the body, so the breadcrumb must not repeat it
    orphan = next(s for s in sections if s.body == "2 Резервные требования")
    assert orphan.prefixes == ("",)


def test_a_nested_childless_heading_keeps_its_ancestors_in_the_breadcrumb():
    """`## 2.1` is the innermost heading with nothing under it, so it becomes the body
    — and `# 2`, which has no body either, still reaches the index through it."""
    markdown = "# 1 Общие\n\nТекст.\n\n# 2 Схема\n\n## 2.1 Порядок согласования\n"
    sections = split_sections(markdown)
    orphan = next(s for s in sections if s.body == "2.1 Порядок согласования")
    assert orphan.prefixes[0] == "2 Схема"
    assert not any(s.body == "2 Схема" for s in sections), "the parent is in the prefix, not twice"


def test_word_bold_runs_come_off_the_breadcrumb():
    """markitdown keeps Word's bold runs: `# **2** **Общая схема...**`."""
    assert clean_heading("**2** **Общая схема документооборота**") == (
        "2 Общая схема документооборота"
    )
    sections = split_sections(SPEC)
    expected = "2 Общая схема документооборота «НСИ и шаблон трансляции»"
    assert any(s.prefixes[0] == expected for s in sections)


def test_slides_split_on_the_slide_marker():
    sections = split_sections(SLIDES)
    labelled = [s for s in sections if s.prefixes[0].startswith("[Slide")]
    assert [s.prefixes[0] for s in labelled] == ["[Slide 1]", "[Slide 2]"]
    assert "Проверка ЗРДС" in labelled[0].body
    assert "Вид операции" in labelled[1].body
    assert "<!-- Slide number:" not in "".join(s.body for s in sections)


def test_spreadsheet_sheet_name_becomes_the_breadcrumb():
    sections = split_sections(SPREADSHEET)
    table = next(s for s in sections if "ВАЛЮТНЫЙ КОНТРОЛЬ" in s.body)
    assert table.prefixes[0] == "ИИ > ДАННЫЕ ПО ПИЛОТУ 2"


def test_oversized_heading_sections_are_split_into_several_parents():
    """A section is what a caller gets back, so it cannot be a whole 40-page chapter."""
    body = "\n\n".join(
        f"Абзац номер {i} про хранение товарно-материальных запасов." for i in range(400)
    )
    sections = split_sections(f"# 1 Раздел\n\n{body}\n\n# 2 Другой\n\nКоротко.")
    first = [s for s in sections if s.prefixes[0] == "1 Раздел"]
    assert len(first) > 1
    assert all(len(s.body) <= 4000 for s in sections)
    assert "".join(s.body for s in first).count("Абзац номер") == 400


def test_sections_never_lose_text():
    """Every word of the document reaches either a section body or a section prefix.

    Heading text moves into the breadcrumb rather than staying in the body, which is
    the one place where sectioning is allowed to move words around."""
    for markdown in (SPEC, TRANSCRIPT, SLIDES, SPREADSHEET, TRANSCRIPT_WITH_SPEAKERS):
        sections = split_sections(markdown, title="T")
        joined = " ".join(s.prefixes[0] + " " + s.body for s in sections)
        dropped = [w for w in markdown.split() if w.strip("*#\\") and w.strip("*#\\") not in joined]
        # only the markup itself may disappear: heading hashes, bold runs, slide markers
        assert all(w.startswith("<!--") or w in ("-->", "number:", "Slide") for w in dropped), (
            markdown[:40],
            dropped,
        )


def test_no_category_can_hand_back_an_unbounded_section():
    """A document with no blank line in it is one structural block, and a section is
    what the caller gets back — so the cap cannot be a heading-path-only rule."""
    wall = " ".join(f"предложение номер {i} про хранение запасов." for i in range(600))
    for markdown in (wall, f"<!-- Slide number: 1 -->\n{wall}\n\n<!-- Slide number: 2 -->\nX"):
        sections = split_sections(markdown)
        assert sections
        assert all(len(s.body) <= 4000 for s in sections), markdown[:40]
    assert " ".join(s.body for s in split_sections(wall)).split() == wall.split()


def test_a_huge_table_is_split_into_sections_that_are_still_tables():
    """The corpus has a 782-row, 65 KB table. As one section it would be handed back
    whole for any hit inside it — and cutting it without repeating the header would
    leave every piece but the first as anonymous columns."""
    header = "| № | Показатель | Периодичность |"
    delimiter = "| --- | --- | --- |"
    rows = [
        f"| {i} | Показатель номер {i} по форме статистической отчетности | месяц |"
        for i in range(200)
    ]
    sections = split_sections("\n".join([header, delimiter, *rows]))
    assert len(sections) > 1
    assert all(len(s.body) <= 4000 for s in sections)
    for s in sections:
        assert s.body.split("\n")[:2] == [header, delimiter]
    kept = [ln for s in sections for ln in s.body.split("\n") if ln not in (header, delimiter)]
    assert kept == rows, "every row survives, exactly once, in order"
