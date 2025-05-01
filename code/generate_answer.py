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
    messages = [
        {
            "role": "system",
            "content":
            f"""
            [Instructions]
            Based on the following context,answer the question accurately and
            concisely. You must not create information you don't see in the
            context, but you **must structure** your response in markdown and not
            only pass the context as answer. Please format your answer in a way
            that is easy to read and visually structured. You also may NOT give
            an explanation to the answer you generated. NEVER start the
            response with 'Based on the provided context'.

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

            [Context]
            {' '.join(retrieved_context)}
            """
        },
        {
            "role": "user",
            "content": f"{prompt}\n\nContexto:\n{retrieved_context}"
        }
    ]
    # Gera tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Gera resposta
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
    # Decodifica a resposta gerada
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in
        zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    # Remove introduções indesejadas da resposta
    intros_indesejadas = [
        "Based on the provided context,",
        "Based on the context provided,"
    ]
    for intro in intros_indesejadas:
        print(intro, response.startswith(intro))
        if response.startswith(intro):
            response = response[len(intro):].lstrip()

    return response
