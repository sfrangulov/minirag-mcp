"""WebVTT transcripts: Teams exports, Zoom exports, and plain captions."""

import pytest

from minirag_mcp.chunker.sections import (
    TIMESTAMP_LINE,
    TRANSCRIPT,
    detect_category,
    split_sections,
)
from minirag_mcp.ingest import vtt
from minirag_mcp.ingest.parser import ParserError, parse_file, supported_extensions

_TEAMS_ID = "0761bb43-3960-4c4d-ac8c-ca8e5fc48078"


def _vtt(*cues: tuple[str, ...], header: str = "WEBVTT") -> str:
    return "\n\n".join([header, *("\n".join(cue) for cue in cues)]) + "\n"


def _teams(utterance: str, start: str, end: str, speaker: str, text: str) -> tuple[str, ...]:
    return (f"{_TEAMS_ID}/{utterance}", f"{start} --> {end}", f"<v {speaker}>{text}</v>")


def _clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}.000"


def _write(tmp_path, name: str, text: str, encoding: str = "utf-8"):
    path = tmp_path / name
    # Teams writes CRLF, so every fixture does too.
    path.write_bytes(text.replace("\n", "\r\n").encode(encoding))
    return path


# The shapes a Teams export has, with invented content: one utterance split into
# segments `17-0`/`17-1` that another speaker's interjection lands between, cue text
# wrapped mid-sentence, and one speaker's separate utterances back to back.
TEAMS = _vtt(
    _teams(
        "17-0",
        "00:00:02.410",
        "00:00:07.880",
        "Marx, Dana",
        "Коллеги, давайте начнём с отчёта\nпо закупкам, у Ольги свежая",
    ),
    _teams("23-0", "00:00:06.300", "00:00:06.900", "Ivanova, Olga", "Ага."),
    _teams("17-1", "00:00:07.880", "00:00:08.420", "Marx, Dana", "выгрузка."),
    _teams(
        "29-0",
        "00:00:27.150",
        "00:00:30.040",
        "Marx, Dana",
        "Цифры по поставщикам сведены в одну таблицу.",
    ),
    _teams(
        "35-0",
        "00:01:12.600",
        "00:01:16.200",
        "Ivanova, Olga",
        "Дубли по контрагентам я уже убрала.",
    ),
    _teams("44-0", "00:02:04.300", "00:02:06.100", "Marx, Dana", "Тогда по срокам оплаты."),
    _teams("45-0", "00:02:09.700", "00:02:11.500", "Marx, Dana", "Там всё совпадает."),
    _teams("51-0", "00:02:38.900", "00:02:40.200", "Ivanova, Olga", "Отлично, спасибо."),
)

TEAMS_MARKDOWN = """**Marx, Dana** 00:02
Коллеги, давайте начнём с отчёта по закупкам, у Ольги свежая выгрузка.

**Ivanova, Olga** 00:06
Ага.

**Marx, Dana** 00:27
Цифры по поставщикам сведены в одну таблицу.

**Ivanova, Olga** 01:12
Дубли по контрагентам я уже убрала.

**Marx, Dana** 02:04
Тогда по срокам оплаты. Там всё совпадает.

**Ivanova, Olga** 02:38
Отлично, спасибо.
"""


def test_vtt_is_a_supported_extension():
    assert ".vtt" in supported_extensions()


def test_teams_cues_become_speaker_turns(tmp_path):
    """Cue ids, end times and milliseconds go; an utterance Teams split into segments
    is whole again, under its first segment's start, even with another speaker's
    interjection between the segments; one speaker's adjacent utterances are one turn."""
    doc = parse_file(_write(tmp_path, "Закупки_ еженедельный отчёт.vtt", TEAMS))
    assert doc.markdown == TEAMS_MARKDOWN
    assert doc.title == "Закупки еженедельный отчёт"


def test_a_teams_vtt_is_windowed_as_a_transcript(tmp_path):
    markdown = parse_file(_write(tmp_path, "call.vtt", TEAMS)).markdown
    assert detect_category(markdown) == TRANSCRIPT
    sections = split_sections(markdown, title="Демо")
    assert [s.prefixes[0] for s in sections] == ["[00:00–02:00] Демо", "[02:00–04:00] Демо"]
    assert "Ivanova, Olga" in sections[1].body


def test_zoom_speaker_prefix_becomes_the_turn_speaker(tmp_path):
    zoom = _vtt(
        ("1", "00:00:01.000 --> 00:00:04.000", "Dana Marx: Начнём с бюджета."),
        ("2", "00:00:05.000 --> 00:00:07.000", "Dana Marx: Он вырос."),
        ("3", "00:00:08.000 --> 00:00:09.000", "Olga Ivanova: Насколько?"),
    )
    markdown = parse_file(_write(tmp_path, "zoom.vtt", zoom)).markdown
    assert markdown == (
        "**Dana Marx** 00:01\nНачнём с бюджета. Он вырос.\n\n**Olga Ivanova** 00:08\nНасколько?\n"
    )


def test_a_colon_in_only_some_cues_is_speech_not_a_speaker(tmp_path):
    captions = _vtt(
        ("1", "00:00:01.000 --> 00:00:02.000", "Итак, повестка: бюджет"),
        ("2", "00:00:03.000 --> 00:00:04.000", "и сроки"),
    )
    markdown = parse_file(_write(tmp_path, "captions.vtt", captions)).markdown
    assert markdown == "00:01\nИтак, повестка: бюджет\n\n00:03\nи сроки\n"


def test_captions_without_speakers_window_as_a_transcript(tmp_path):
    """A caption track has no speakers and opens with metadata lines in its header
    block; every cue keeps its own timestamp line."""
    cues = [
        (f"00:{minute:02d}:{second:02d}.000 --> 00:{minute:02d}:{second + 2:02d}.000", word)
        for minute, second, word in [
            (0, 1, "раз"),
            (0, 30, "два"),
            (1, 0, "три"),
            (2, 5, "четыре"),
            (2, 40, "пять"),
        ]
    ]
    text = _vtt(*cues, header="WEBVTT\nKind: captions\nLanguage: ru")
    markdown = parse_file(_write(tmp_path, "lecture.vtt", text)).markdown
    assert markdown.startswith("00:01\nраз\n\n00:30\nдва\n")
    assert detect_category(markdown) == TRANSCRIPT
    assert len(split_sections(markdown, title="Лекция")) == 2


def test_a_voice_tag_needs_no_closing_tag(tmp_path):
    text = _vtt(("00:00:01.000 --> 00:00:02.000", "<v Dana Marx>Привет"))
    markdown = parse_file(_write(tmp_path, "open.vtt", text)).markdown
    assert markdown == "**Dana Marx** 00:01\nПривет\n"


def test_inline_tags_are_dropped_and_escapes_decoded(tmp_path):
    """Markup goes before entities are decoded, so text that only spells a tag stays."""
    text = _vtt(
        (
            "00:00:01.000 --> 00:00:02.000",
            "<c.yellow>Цена</c> &lt;b&gt; <00:00:01.500><i>важна</i> &amp; срочна",
        )
    )
    markdown = parse_file(_write(tmp_path, "tags.vtt", text)).markdown
    assert markdown == "00:01\nЦена <b> важна & срочна\n"


def test_timestamps_use_the_transcripts_own_notation(tmp_path):
    """Hours are optional in WebVTT; past the hour the label grows one, as the window
    labels do."""
    text = _vtt(
        ("02:05.000 --> 02:07.000", "<v A>до часа"),
        ("01:02:03.900 --> 01:02:05.000", "<v B>после часа"),
    )
    markdown = parse_file(_write(tmp_path, "long.vtt", text)).markdown
    assert markdown == "**A** 02:05\nдо часа\n\n**B** 1:02:03\nпосле часа\n"


def test_every_turn_header_is_one_the_sectioner_recognises(tmp_path):
    """A header the transcript pattern does not match is a turn silently folded into
    the previous window. A speaker that cannot fit the bolded form keeps its words in
    the turn's text instead."""
    long_name = "Очень длинное имя " * 6
    text = _vtt(
        ("00:00:01.000 --> 00:00:02.000", "<v Dana **the PM**>первая"),
        ("00:00:03.000 --> 00:00:04.000", f"<v {long_name.strip()}>вторая"),
    )
    markdown = parse_file(_write(tmp_path, "names.vtt", text)).markdown
    turns = markdown.strip().split("\n\n")
    assert len(turns) == 2
    assert all(TIMESTAMP_LINE.match(turn.split("\n")[0]) for turn in turns)
    assert long_name.strip() in turns[1]


def test_a_note_block_is_kept_as_a_quote(tmp_path):
    text = _vtt(
        ("NOTE Встреча: квартальный обзор",),
        ("00:00:01.000 --> 00:00:02.000", "<v A>Начнём"),
    )
    markdown = parse_file(_write(tmp_path, "note.vtt", text)).markdown
    assert markdown == "> Встреча: квартальный обзор\n\n**A** 00:01\nНачнём\n"


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_a_byte_order_mark_picks_the_encoding(tmp_path, encoding):
    """UTF-32-LE's mark begins with UTF-16-LE's, so the order the marks are tried in
    decides which of the two a UTF-32 file is read as."""
    text = _vtt(("00:00:01.000 --> 00:00:02.000", "<v A>Да"))
    assert parse_file(_write(tmp_path, "bom.vtt", text, encoding=encoding)).markdown == (
        "**A** 00:01\nДа\n"
    )


def test_a_cp1251_file_decodes(tmp_path):
    text = _vtt(
        (
            "00:00:01.000 --> 00:00:04.000",
            "<v Иванова, Ольга>Согласовали бюджет на следующий квартал.",
        ),
        ("00:00:05.000 --> 00:00:09.000", "<v Маркс, Дана>Тогда переходим к срокам поставки."),
    )
    markdown = parse_file(_write(tmp_path, "legacy.vtt", text, encoding="cp1251")).markdown
    assert "**Иванова, Ольга** 00:01\nСогласовали бюджет" in markdown


def test_a_malformed_cue_fails_the_file_instead_of_indexing_raw_text(tmp_path):
    """Nothing falls back to markitdown: its plain-text converter accepts a .vtt and
    would index the raw cue ids, arrows and tags as a clean success."""
    text = _vtt(("00:00:01.000 --> 00:00:02.000", "<v A>Да"), ("просто строка", "и ещё одна"))
    with pytest.raises(ParserError, match="broken.vtt"):
        parse_file(_write(tmp_path, "broken.vtt", text))


def test_a_monologue_still_windows_every_two_minutes(tmp_path):
    """A presenter's ten minutes are one speaker's adjacent utterances throughout.
    Merged without a limit they were one turn under its first timestamp, and the file
    stopped being a transcript at all."""
    demo = _vtt(
        *(
            _teams(f"{n}-0", _clock(n * 20), _clock(n * 20 + 15), "Marx, Dana", f"Слайд {n}.")
            for n in range(30)
        )
    )
    markdown = parse_file(_write(tmp_path, "demo.vtt", demo)).markdown
    assert detect_category(markdown) == TRANSCRIPT
    assert [s.prefixes[1] for s in split_sections(markdown, title="Демо")] == [
        "[00:00–02:00]",
        "[02:00–04:00]",
        "[04:00–06:00]",
        "[06:00–08:00]",
        "[08:00–10:00]",
    ]


def test_an_encoding_too_short_to_tell_fails_instead_of_indexing_mojibake(tmp_path):
    """On one short cp1251 cue charset detection names an unrelated code page and
    decodes the file to nonsense without complaint."""
    text = _vtt(("00:00:01.000 --> 00:00:02.000", "<v Ольга>Да."))
    with pytest.raises(ParserError, match="UTF-8"):
        parse_file(_write(tmp_path, "short.vtt", text, encoding="cp1251"))


@pytest.mark.parametrize("identifier", [(), ("1",)])
def test_a_cue_right_under_the_header_is_kept(tmp_path, identifier):
    first = ["WEBVTT", *identifier, "00:00:01.000 --> 00:00:02.000", "<v A>Первая"]
    text = "\n".join(first) + "\n\n00:00:03.000 --> 00:00:04.000\n<v B>Вторая\n"
    assert parse_file(_write(tmp_path, "tight.vtt", text)).markdown == (
        "**A** 00:01\nПервая\n\n**B** 00:03\nВторая\n"
    )


def test_a_bare_angle_bracket_is_text_not_a_tag(tmp_path):
    text = _vtt(("00:00:01.000 --> 00:00:02.000", "<v A>ставка <5% и >3% годовых"))
    assert parse_file(_write(tmp_path, "rates.vtt", text)).markdown == (
        "**A** 00:01\nставка <5% и >3% годовых\n"
    )


def test_unpadded_timings_of_older_teams_exports_parse(tmp_path):
    text = _vtt(("0:0:5.120 --> 0:0:9.0", "<v A>Начали"), ("0:2:7.5 --> 0:2:9.0", "<v B>Дальше"))
    assert parse_file(_write(tmp_path, "old.vtt", text)).markdown == (
        "**A** 00:05\nНачали\n\n**B** 02:07\nДальше\n"
    )


def test_a_clock_time_is_not_a_speaker(tmp_path):
    captions = _vtt(
        ("1", "00:00:01.000 --> 00:00:02.000", "Начало в 10:30"),
        ("2", "00:00:03.000 --> 00:00:04.000", "Конец в 12:00"),
    )
    markdown = parse_file(_write(tmp_path, "times.vtt", captions)).markdown
    assert markdown == "00:01\nНачало в 10:30\n\n00:03\nКонец в 12:00\n"


def test_the_longest_name_a_header_takes_is_one_the_sectioner_takes(tmp_path):
    """The limit on a bolded name is the sectioner's, restated: were the sectioner's
    to shrink, a name between the two would silently stop opening a turn."""
    fits, too_long = "Я" * vtt._MAX_SPEAKER, "Я" * (vtt._MAX_SPEAKER + 1)
    text = _vtt(
        ("00:00:01.000 --> 00:00:02.000", f"<v {fits}>первая"),
        ("00:00:03.000 --> 00:00:04.000", f"<v {too_long}>вторая"),
    )
    first, second = parse_file(_write(tmp_path, "n.vtt", text)).markdown.strip().split("\n\n")
    assert first.split("\n")[0] == f"**{fits}** 00:01"
    assert TIMESTAMP_LINE.match(first.split("\n")[0])
    assert second == f"00:03\n{too_long}: вторая"


def test_style_and_region_blocks_are_not_text(tmp_path):
    text = _vtt(
        ("STYLE", "::cue { color: red }"),
        ("REGION", "id:speaker width:40%"),
        ("00:00:01.000 --> 00:00:02.000", "<v A>Да"),
    )
    assert parse_file(_write(tmp_path, "style.vtt", text)).markdown == "**A** 00:01\nДа\n"


def test_a_teams_segment_in_another_voice_starts_its_own_turn(tmp_path):
    text = _vtt(
        _teams("8-0", "00:00:01.000", "00:00:02.000", "Marx, Dana", "Первая часть"),
        _teams("8-1", "00:00:02.000", "00:00:03.000", "Ivanova, Olga", "чужая реплика"),
    )
    assert parse_file(_write(tmp_path, "voices.vtt", text)).markdown == (
        "**Marx, Dana** 00:01\nПервая часть\n\n**Ivanova, Olga** 00:02\nчужая реплика\n"
    )


def test_a_list_is_neither_merged_nor_reflowed(tmp_path):
    text = _vtt(
        ("00:00:01.000 --> 00:00:02.000", "<v A>Повестка:"),
        ("00:00:03.000 --> 00:00:04.000", "<v A>1. бюджет\n2. сроки"),
    )
    assert parse_file(_write(tmp_path, "list.vtt", text)).markdown == (
        "**A** 00:01\nПовестка:\n\n**A** 00:03\n1. бюджет\n2. сроки\n"
    )
