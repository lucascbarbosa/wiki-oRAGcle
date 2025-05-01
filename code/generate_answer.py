"""03__generate_answer."""
import pandas as pd
import torch


def retrieve_context(
    embedding_model,
    faiss_index,
    processed_pages_df: pd.DataFrame,
    prompt: str,
    k: int) -> list:
    """Retrieve context related to prompt."""
    # Gera embedding da prompt
    prompt_embedding = embedding_model.encode([prompt]).astype('float32')

    # Busca os k textos mais relevantes
    distances, indices = faiss_index.search(prompt_embedding, k)

    # Recupera os textos relevantes com base nos índices
    retrieved_context = [
        processed_pages_df.iloc[i]['text'] for i in indices[0]]

    return retrieved_context


def generate_answer(device, model, tokenizer, prompt: str, retrieved_context: list):
    """Generate answer with llm and retrieved context."""
    # Formata contexto
    prompt_with_context = f"""
        Based on the following context, answer the question accurately and
        concisely. You must not create information you don't see in the
        context.

        [Question]
        {prompt}

        [Context]
        {' '.join(retrieved_context)}

        [Answer]
    """
    # Gera tokens
    inputs = tokenizer(prompt_with_context, return_tensors="pt").to(device)

    # Gera resposta
    with torch.inference_mode():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=200,
            do_sample=False,
        )

    # Decodifica a resposta gerada
    answer = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).split('[Answer]\n')[1].strip()

    return answer
