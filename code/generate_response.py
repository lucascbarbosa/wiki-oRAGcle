"""Script for response generation."""
import pandas as pd
import google.generativeai as genai


def retrieve_context(
    embedding_model,
    faiss_index,
    processed_pages_df: pd.DataFrame,
    question: str,
    k: int) -> list:
    """Retrieve context related to question."""
    # Gera embedding da question
    question_embedding = embedding_model.encode([question]).astype('float32')

    # Busca os k textos mais relevantes
    distances, indices = faiss_index.search(question_embedding, k)

    # Recupera os textos relevantes com base nos índices
    retrieved_context = [
        processed_pages_df.iloc[i]['text'] for i in indices[0]]

    return retrieved_context


def generate_response(
    gemini_chat,
    question: str,
    retrieved_context: list,
    temperature: float = None,
) -> str:
    """Generate answer with Gemini API and retrieved context."""
    prompt = f"""
            [Instructions]
            Based on the following context,answer the question accurately.

            You must not create information you don't see in the context.

            You also may NOT give an explanation to the answer you
            generated.

            NEVER start the response with 'Based on the provided
            context'.

            ALWAYS avoid listing phrases in a enumerated or bulletized list.

            The response must be COMPLETE, but also DIRECT. Don't include
            additional facts that don't contribute to answering the question.

            Example:
            User: Who is Jon Snow?
            Answer: **Jon Snow** is the bastard son of Eddard Stark, Lord of
            Winterfell. He has five half-siblings: Robb, Sansa, Arya, Bran, and
            Rickon Stark. Unaware of the identity of his mother, Jon was raised at
            Winterfell. At the age of fourteen, Jon joins the Night's Watch, where
            he earns the nickname Lord Snow. Jon is one of the major POV characters
            in *A Song of Ice and Fire*.

            [Context]
            {' '.join(retrieved_context)}

            [Question]
            {question}
        """

    # Configuration for generation
    config_kwargs = {}
    if temperature is not None:
        config_kwargs['temperature'] = temperature
    generation_config = genai.types.GenerationConfig(**config_kwargs)

    # Generate response
    response = gemini_chat.send_message(
        content=prompt,
        generation_config=generation_config,
    )

    return response.text.strip()
