"""03__generate_answer."""
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "artifacts/faiss_index.index"
LLM_NAME = "allenai/OLMo-2-1124-7B-Instruct"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME)


def retrieve_context(prompt: str, k: int) -> list:
    """Retrieve context related to prompt."""
    # Gera embedding da prompt
    prompt_embedding = embedding_model.encode([prompt]).astype('float32')

    # Busca os k textos mais relevantes
    distances, indices = faiss_index.search(prompt_embedding, k)

    # Recupera os textos relevantes com base nos índices
    retrieved_context = [
        processed_pages_df.iloc[i]['text'] for i in indices[0]]

    return retrieved_context


def generate_answer(prompt: str, retrieved_context: list):
    """Generate answer with llm and retrieved context."""
    # Formata contexto
    context = (
        f"Question: {prompt}\nContext: "
        f"{' '.join(retrieved_context)}\nAnswer:"
    )

    # Gera tokens
    inputs = tokenizer(context, return_tensors="pt")

    # Gera resposta
    outputs = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_return_sequences=1,
        temperature=0.7
    )

    # Decodifica a resposta gerada
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
