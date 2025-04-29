"""03__generate_answer."""
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# Carrega bases e modelos
DATABASE_PATH = "artifacts/database.parquet"
FAISS_INDEX_PATH = "artifacts/faiss_index.index"
LLM_NAME = "mistralai/Mistral-7B-v0.1"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
pages_df = pd.read_parquet(DATABASE_PATH)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME, device_map="auto", torch_dtype="auto")


def retrieve_context(prompt: str, k: int) -> list:
    """Retrieve context related to prompt."""
    # Gera embedding da prompt
    prompt_embedding = embedding_model.encode([prompt]).astype('float32')

    # Busca os k textos mais relevantes
    distances, indices = faiss_index.search(prompt_embedding, k)

    # Recupera os textos relevantes com base nos índices
    retrieved_context = [pages_df.iloc[i]['content'] for i in indices[0]]

    return retrieved_context


def generate_answer_with_llama(query, retrieved_context):
    """Generate answer with llama model and retrieved context."""
    # Formata contexto
    context = (
        f"Question: {query}\nContext: "
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


def process_prompt(prompt, k=1):
    """Process prompt with context."""
    # Gera contexto
    retrieved_texts = retrieve_context(prompt, k)

    # Gera resposta
    answer = generate_answer_with_llama(prompt, retrieved_texts)
    return answer


# Exemplo de uso
prompt = "Quem é Jon Snow?"
answer = process_prompt(prompt, k=1)
print("Resposta gerada:", answer)
