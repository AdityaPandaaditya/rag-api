import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.llm import embed_query, generate_answer
from app.models import Chunk, Document
from app.worker import process_document

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data/uploads"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="PDF Question Answering API", lifespan=lifespan)


@app.post("/documents", status_code=202)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    doc = Document(filename=file.filename, status="pending")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / f"{doc.id}.pdf").write_bytes(data)

    process_document.delay(doc.id)
    return {"id": doc.id, "filename": doc.filename, "status": doc.status}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    rows = db.execute(
        select(
            Document.id,
            Document.filename,
            Document.created_at,
            Document.status,
            Document.error,
            func.count(Chunk.id).label("chunks"),
        )
        .join(Chunk, isouter=True)
        .group_by(Document.id)
        .order_by(Document.id)
    ).all()
    return [dict(r._mapping) for r in rows]


class AskRequest(BaseModel):
    question: str
    document_id: int | None = None
    top_k: int = 5


@app.post("/ask")
async def ask(req: AskRequest, db: Session = Depends(get_db)):
    try:
        q_vec = await run_in_threadpool(embed_query, req.question)
    except Exception as e:
        raise HTTPException(502, f"Embedding failed: {e}")

    stmt = (
        select(Chunk)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(q_vec))
        .limit(req.top_k)
    )
    if req.document_id is not None:
        stmt = stmt.where(Chunk.document_id == req.document_id)
    chunks = db.scalars(stmt).all()
    if not chunks:
        raise HTTPException(404, "No indexed chunks found. Upload a PDF first.")

    try:
        answer = await run_in_threadpool(generate_answer, req.question, chunks)
    except Exception as e:
        raise HTTPException(502, f"Generation failed: {e}")

    return {
        "answer": answer,
        "sources": [
            {"page": c.page, "chunk_id": c.id, "preview": c.content[:150]}
            for c in chunks
        ],
    }