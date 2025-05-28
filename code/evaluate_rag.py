"""Script for RAG evaluation."""
import faiss
import google.generativeai as genai
import pandas as pd
from generate_response import generate_response, retrieve_context
from sentence_transformers import SentenceTransformer


# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "../artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = "AIzaSyAV3VJG9STCErIBXz1LNls0V3SQ_UVi24U"
genai.configure(api_key=GEMINI_API_KEY)

# Setup variables
print("\nSetting up models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
gemini_client = genai.GenerativeModel(model_name=GEMINI_MODEL)
gemini_chat = gemini_client.start_chat(history=[])

# Read QA benchmark data
benchmark_data = pd.read_json("../artifacts/benchmark.json")
scores = {}
for subject in benchmark_data.columns:
    subject_qas = benchmark_data[subject]
    predictions = []
    references = []
    for qa in subject_qas:
        question = qa['question']
        response_ref = qa['answer']

        print(
            f"Question: {question}\n"
            f"Reference: {response_ref}\n"
        )

        retrieved_context = retrieve_context(
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            processed_pages_df=processed_pages_df,
            question=question,
            k=10
        )
        response_pred = generate_response(
            gemini_chat=gemini_chat,
            question=question,
            retrieved_context=retrieved_context,
            max_tokens=512,
            temperature=0.7
        )

        print(
            f"Response: {response_pred}\n\n"
        )

        references.append(response_ref)
        predictions.append(response_pred)

    # Evaluate metrics
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(
        predictions=predictions, references=references)['bleu']
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(
        predictions=predictions, references=references)
    rouge_score = (
        (
            rouge_score['rouge1'] + rouge_score['rouge2'] +
            rouge_score['rougeL'] + rouge_score['rougeL']
        ) / 4
    )
    scores[subject] = {
        'bleu': bleu_score,
        'rouge': float(rouge_score),
    }
    print(f'\n{subject}: {scores[subject]}\n')
