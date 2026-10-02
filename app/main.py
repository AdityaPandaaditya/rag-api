from contextlib import asynccontextmanager

from fastapi.concurrency import run_in_threadpool
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.llm import embed_documents
from app.ingest import chunk_text, extract_pages
from app.models import Chunk, Document

from app.llm import embed_documents, embed_query, generate_answer
from pydantic import BaseModel

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="PDF Question Answering API", lifespan=lifespan)


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    pages = extract_pages(data)
    if not pages:
        raise HTTPException(422, "No extractable text found (scanned PDFs are not supported)")

    doc = Document(filename=file.filename)
    for page_no, text in pages:
        for idx, piece in enumerate(chunk_text(text)):
            doc.chunks.append(Chunk(page=page_no, chunk_index=idx, content=piece))
    try:
        vectors = await run_in_threadpool(embed_documents, [c.content for c in doc.chunks])
    except Exception as e:
        raise HTTPException(502, f"Embedding failed: {e}")
    for chunk, vec in zip(doc.chunks, vectors):
        chunk.embedding = vec
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"id": doc.id, "filename": doc.filename, "pages": len(pages), "chunks": len(doc.chunks)}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Document.id, Document.filename, Document.created_at, func.count(Chunk.id))
        .join(Chunk, isouter=True)
        .group_by(Document.id)
        .order_by(Document.id)
    ).all()
    return [{"id": r[0], "filename": r[1], "created_at": r[2], "chunks": r[3]} for r in rows]




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
