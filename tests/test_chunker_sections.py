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


def test_transcript_prefix_degrades_to_the_timestamp_alone():
    """A long meeting title can be most of the budget; the label must survive it."""
    sections = split_sections(TRANSCRIPT, title="x" * 400)
    window = next(s for s in sections if s.prefixes[0].startswith("["))
    assert window.prefixes[1] == "[00:00–02:00]"
    assert window.prefixes[-1] == ""


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
    every child carries it in the breadcrumb."""
    sections = split_sections(SPEC)
    assert not any(s.body.strip() == "" for s in sections)
    assert all("1 Общие положения" in s.prefixes[0] for s in sections[:1])


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
