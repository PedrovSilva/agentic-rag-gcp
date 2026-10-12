from google import genai
from google.genai import types
from app.core.config import settings

class GenAIProvider:
    def __init__(self):
        kwargs = {}
        if settings.google_cloud_project:
            kwargs = {
                "vertexai": settings.vertex_ai,
                "project": settings.google_cloud_project,
                "location": settings.google_cloud_location,
            }
        self.client = genai.Client(**kwargs)

    def embed(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
        )
        return response.embeddings[0].values

    def generate(self, question: str, context: str) -> str:
        prompt = f"""You are a technical documentation assistant.
Answer ONLY from the supplied context. If the context is insufficient, say that you do not have enough information.
Cite sources using [source]. Be concise and factual.

CONTEXT:
{context}

QUESTION:
{question}
"""
        response = self.client.models.generate_content(
            model=settings.generation_model,
            contents=prompt,
        )
        return response.text or "No answer generated."
