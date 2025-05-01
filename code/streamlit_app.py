"""04__streamlit_app."""
import streamlit
from generate_answer import generate_answer, retrieve_context


def process_prompt(prompt: str, k: int):
    """Process prompt with context."""
    # Gera contexto
    retrieved_context = retrieve_context(prompt, k)

    # Gera resposta
    answer = generate_answer(prompt, retrieved_context)
    return answer


# Exemplo de uso
prompt = "Quem é Jon Snow?"
answer = process_prompt(prompt, k=5)
print("Resposta gerada:", answer)
