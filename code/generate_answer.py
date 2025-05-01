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


def generate_answer(
    device,
    model,
    tokenizer,
    prompt: str,
    retrieved_context: list,
    max_tokens: int) -> str:
    """Generate answer with llm and retrieved context."""
    # Formata contexto
    prompt_with_context = f"""
        [Instructions]
        Based on the following context,answer the question accurately and
        concisely. You must not create information you don't see in the
        context, but you **must structure** your response in markdown and not
        only pass the context as answer. Please format your answer in a way
        that is easy to read and visually structured. You also may NOT give
        an explanation to the answer you generated.

        Example:
        User: Who is Jon Snow?
        Answer: **Jon Snow** is the bastard son of Eddard Stark, Lord of
        Winterfell. He has five half-siblings: Robb, Sansa, Arya, Bran, and
        Rickon Stark. Unaware of the identity of his mother, Jon was raised at
        Winterfell. At the age of fourteen, Jon joins the Night's Watch, where
        he earns the nickname Lord Snow. Jon is one of the major POV characters
        in *A Song of Ice and Fire*.

        ## Appearance and Character
        Jon has the long face of the Starks [...]

        [Question]
        {prompt}

        [Context]
        {' '.join(retrieved_context)}

        [Answer]
    """
    # Gera tokens
    # tokenizer.pad_token_id = tokenizer.unk_token_id
    inputs = tokenizer(prompt_with_context, return_tensors="pt").to(device)

    # Gera resposta
    with torch.inference_mode():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_tokens,
        )

    # Decodifica a resposta gerada
    answer = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )[len(prompt_with_context):]

    return answer
