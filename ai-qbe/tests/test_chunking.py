from rag.chunking import clean_text, semantic_chunk


def test_clean_text_strips_page_number_lines():
    text = "Some content\n42\nMore content"
    cleaned = clean_text(text)
    assert "42" not in cleaned.split("\n")


def test_clean_text_collapses_whitespace():
    text = "Too    many     spaces"
    assert clean_text(text) == "Too many spaces"


def test_clean_text_collapses_excess_blank_lines():
    text = "Paragraph one\n\n\n\n\nParagraph two"
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned


def test_semantic_chunk_empty_text_returns_no_chunks():
    assert semantic_chunk("") == []
    assert semantic_chunk("   \n\n  ") == []


def test_semantic_chunk_single_short_paragraph_is_one_chunk():
    text = "Force equals mass times acceleration, a foundational law of classical mechanics."
    chunks = semantic_chunk(text, page=3)
    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert "Force equals mass" in chunks[0].text


def test_semantic_chunk_respects_heading_boundaries():
    text = "Newton's Second Law\n\nForce equals mass times acceleration.\n\nThird Law\n\nEvery action has an equal and opposite reaction."
    chunks = semantic_chunk(text, min_words=0)
    sections = {c.section for c in chunks}
    assert "Newton's Second Law" in sections
    assert "Third Law" in sections


def test_semantic_chunk_packs_multiple_paragraphs_under_max_words():
    paragraphs = "\n\n".join(f"This is paragraph number {i} with a few words in it." for i in range(20))
    chunks = semantic_chunk(paragraphs, max_words=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.word_count <= 60  # some slack for packing boundary


def test_semantic_chunk_splits_an_overlong_single_paragraph():
    long_paragraph = " ".join(f"word{i}." for i in range(500))
    chunks = semantic_chunk(long_paragraph, max_words=100)
    assert len(chunks) > 1
    assert all(c.word_count <= 110 for c in chunks)


def test_semantic_chunk_merges_undersized_trailing_chunk():
    text = "Heading\n\n" + " ".join(f"word{i}" for i in range(50)) + "\n\nshort trailer"
    chunks = semantic_chunk(text, max_words=60, min_words=10)
    # the short trailer should be merged into the previous chunk, not stand alone
    assert all(c.word_count >= 10 or len(chunks) == 1 for c in chunks)


def test_semantic_chunk_indices_are_sequential():
    text = "\n\n".join(f"Paragraph {i} " + " ".join(["word"] * 60) for i in range(5))
    chunks = semantic_chunk(text, max_words=50)
    assert [c.index for c in chunks] == list(range(len(chunks)))
