"""03__generate_answer."""
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch_dtype = torch.float16

# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "artifacts/faiss_index.index"
LLM_NAME = "allenai/OLMo-2-0425-1B-Instruct"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME, torch_dtype=torch_dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)


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
            max_new_tokens=300,
            do_sample=False,
        )

    # Decodifica a resposta gerada
    answer = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).split('[Answer]\n')[1].strip()

    return answer


# def process_prompt(prompt: str, k: int):
#     """Process prompt with context."""
#     # Gera contexto
#     retrieved_context = retrieve_context(prompt, k)

#     # Gera resposta
#     answer = generate_answer(prompt, retrieved_context)
#     return answer


# # Exemplo de uso
# prompt = "Who is Jon Snow?"
# answer = process_prompt(prompt, k=5)
# print("Resposta gerada:", answer)
