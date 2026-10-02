from io import BytesIO

from pypdf import PdfReader


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has text. Pages are 1-based."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # collapse whitespace
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into ~size-character chunks that overlap, breaking at spaces
    so words are not cut in half. Overlap keeps context that spans a boundary."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        new_start = max(end - overlap, start + 1)
        # If we landed mid-word, skip forward to the next word
        if text[new_start - 1] != " ":
            next_space = text.find(" ", new_start, end)
            if next_space != -1:
                new_start = next_space + 1
            else:
                new_start = end  # no space in overlap: skip overlap rather than cut a word
        start = new_start
    return [c for c in chunks if c]
