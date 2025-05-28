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
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    """Generate answer with Gemini API and retrieved context."""
    prompt = f"""
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

            [Question]
            {question}
        """

    # Configuration for generation
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    # Generate response
    response = gemini_chat.send_message(
        content=prompt,
        generation_config=generation_config
    )

    return response.text.strip()
