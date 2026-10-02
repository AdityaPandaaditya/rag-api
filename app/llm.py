import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.models import EMBEDDING_DIM

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.1-flash-lite"

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 100):  # max 100 texts per API call
        result = _client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 100],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        vectors.extend(e.values for e in result.embeddings)
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embed([text], "RETRIEVAL_QUERY")[0]

def generate_answer(question: str, chunks) -> str:
    context = "\n\n".join(f"[page {c.page}] {c.content}" for c in chunks)
    prompt = f"""Answer the question using ONLY the context below.
        Cite the pages you used, like (page 3).
        If the context doesn't contain the answer, say "I couldn't find that in the document."

        Context:
        {context}

        Question: {question}"""
    response = _client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text