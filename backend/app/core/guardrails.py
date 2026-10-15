import unicodedata

MAX_QUESTION_LENGTH = 4000

def normalize_question(question: str) -> str:
    question = unicodedata.normalize("NFKC", question).strip()

    if not question:
        raise ValueError("A pergunta não pode estar vazia.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"A pergunta deve ter no máximo {MAX_QUESTION_LENGTH} caracteres.")

    if any(
        unicodedata.category(char) == "Cc"
        and char not in "\n\r\t"
        for char in question
    ):
        raise ValueError("A pergunta contém caracteres de controle inválidos.")

    return question

def select_relevant_sources(
    sources: list[dict],
    max_distance: float,
) -> list[dict]:
    """Filtra resultados por distância de cosseno."""
    return [
        source
        for source in sources
        if isinstance(source.get("distance"), (int, float))
        and 0 <= source["distance"] <= max_distance
    ]