"""Genuine excerpts from the 558-document corpus the chunker was designed against.

Every string here was read out of a real indexed document, unedited. Synthetic
fixtures were what let the old chunker look correct while cutting xlsx rows in half
and treating a 90-character transcript turn as a retrieval unit: the shapes that
break a chunker — markitdown's bolded numbered headings, `NaN` cells, an empty
header row, a speaker prefix before the timestamp — are not shapes anyone invents.
"""

# instructions/12. МСФО/И-68_МСФО_НСИ и шаблон трансляции_v2_01.12.2025.docx
# Numbered headings with Word's bold runs still in them, and an in-spec table.
SPEC = """# **1 Общие положения**

## 1.1 Глоссарий

| Термин / Сокращение | Определение |
| --- | --- |
| БП | Бизнес-процесс |
| ВНА | Внеоборотные активы |
| ИС | Информационная база |
| НСИ | Нормативно-справочная информация |
| ОС | Основное средство |
| БС | Банковский счет |

# **2** **Общая схема документооборота «НСИ и шаблон трансляции»**

|  |  |  |
| --- | --- | --- |
| **№** | **Операция (Шаг БП)** | **Объект 1С** |
| 3.1.1 | Отражение ставки капитализации затрат на вскрышу | Документ «Установка коэффициента капитализации затрат на вскрышу» |
| 3.2.1 | Ведение справочника «Типы банковских счетов для ноты МСФО» | Справочник «Типы банковских счетов для ноты МСФО» |
"""

# transcripts/010425 Кросс-функциональное демо-ru-RU.docx
# The common shape: a bare timestamp alone on its line, then the speech.
TRANSCRIPT = """# 010425 Кросс-функциональное демо-ru-RU

**010425 Кросс-функциональное демо**

0:02
Угу, давайте небольшую воду, заткнись небольшую Сергей, Добрый день.

0:08
Добрый день Сергей.

0:11
А в чём повестка нашей сегодняшней встречи? Ну, по теме и и повестке описанных приглашений вы, наверное, уже увидели, что основная основные моменты, которые мы хотим сегодня подсветить, обсудить.

1:06
Текущую встречу не нужно, а пройдёмся скорее по вопросам уже предметно точечно.

2:05
Так видно.

2:08
Да, все видно отлично.

3:40
Возвратные отходы и побочный выпуск здесь у нас будет 2 спикера.
"""
TRANSCRIPT_TITLE = "010425 Кросс-функциональное демо"

# ai_initiatives/transcripts/Воркшоп_2___Валютный_контроль.docx
# The other shape: Teams keeps the speaker, so the timestamp is not alone on its line.
# Two of the corpus's 107 transcripts look like this, and both were missed by a
# pattern that only accepted a bare timestamp.
TRANSCRIPT_WITH_SPEAKERS = """# Воркшоп 2 Валютный контроль

**Воркшоп 2 — Валютный контроль-20260806\\_150141-Meeting Recording**

August 6, 2026, 11:01AM

30m 33s

**Skachkov, Dmitry** started transcription

**Dana Marx** 0:03
Документы, кому куда мы должны были отправить?

**Skachkov, Dmitry** 0:05
Угу.

**Dana Marx** 0:07
Или там, как мы договорились, за шерим доступ или по почте отправим угу.

**Frangulov, Sergei** 0:09
Да да, я подготовил уже это.
Я отправлю там сразу после звонка.

**Dana Marx** 2:41
Хорошо, спасибо.
"""

# ai_initiatives/transcripts/ЗРДС_Пилот1_Проверка_ЗРДС_2.pptx
SLIDES = """# ЗРДС Пилот1 Проверка ЗРДС 2

<!-- Slide number: 1 -->
Пилот 1
Проверка ЗРДС

Регламент проверки, типы платежей и 11 примеров обработанных заявок,
включая случаи расхождений

### Notes:

<!-- Slide number: 2 -->
Регламент проверки ЗРДС
Пошаговая проверка заявки на оплату и 14 параметров сверки

Получение документов
Открытие 1С УХ
Outlook — документы по предприятию, поступившие на оплату
Банк-клиент → «Валютный контроль»: договор и остаток по контракту
| № | Параметр сверки |
| --- | --- |
| 1 | Вид операции |
"""

# ai_initiatives/transcripts/ИИ.xlsx — markitdown names a sheet with an H2 and emits
# pandas' `Unnamed:` header and `NaN` cells verbatim.
SPREADSHEET = """# ИИ

## ДАННЫЕ ПО ПИЛОТУ 2

| Unnamed: 0 | Unnamed: 1 | Unnamed: 2 |
| --- | --- | --- |
| NaN | ДАННЫЕ ПО ПИЛОТУ 2 «ВАЛЮТНЫЙ КОНТРОЛЬ» | NaN |
| NaN | NaN | NaN |
| 1.0 | Актуальный пакет нормативных документов РК и МФЦ по валютному контролю (законы, постановления, инструкции) | NaN |
| 2.0 | Внутренние регламенты/чек-листы команды, если формализованы | NaN |
| 3.0 | 5–10 примеров договоров/описаний сделок разной сложности с результатами проверки | NaN |
| 4.0 | Контакт эксперта по валютному контролю — валидация логики чек-листа (УНК, сроки репатриации, банки, закрывающие документы) | NaN |
"""


def words(text: str) -> list[str]:
    """Every non-whitespace token, for proving a split lost nothing."""
    return text.split()
