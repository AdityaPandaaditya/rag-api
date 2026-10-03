import os
from pathlib import Path

from celery import Celery

from app.db import SessionLocal
from app.ingest import chunk_text, extract_pages
from app.llm import embed_documents
from app.models import Chunk, Document

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data/uploads"))

celery_app = Celery("rag", broker=REDIS_URL, backend=REDIS_URL)


@celery_app.task
def ping(name: str) -> str:
    return f"pong, {name}"


@celery_app.task
def process_document(doc_id: int) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        doc.status = "processing"
        db.commit()

        pdf_bytes = (UPLOAD_DIR / f"{doc_id}.pdf").read_bytes()
        pages = extract_pages(pdf_bytes)
        if not pages:
            raise ValueError("No extractable text found (scanned PDFs are not supported)")

        for page_no, text in pages:
            for idx, piece in enumerate(chunk_text(text)):
                doc.chunks.append(Chunk(page=page_no, chunk_index=idx, content=piece))

        vectors = embed_documents([c.content for c in doc.chunks])
        for chunk, vec in zip(doc.chunks, vectors):
            chunk.embedding = vec

        doc.status = "ready"
        db.commit()
    except Exception as e:
        db.rollback()
        doc = db.get(Document, doc_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = str(e)[:500]
            db.commit()
    finally:
        db.close()