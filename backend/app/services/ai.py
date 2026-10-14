from functools import lru_cache
from math import sqrt
from typing import Any

from groq import Groq
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt


@lru_cache
def get_groq_client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()

    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector.tolist()


def generate_answer(question: str, context: str) -> str:
    settings = get_settings()

    completion = get_groq_client().chat.completions.create(
        model=settings.groq_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(question, context),
            },
        ],
    )

    answer = completion.choices[0].message.content or ""
    return answer.strip()
