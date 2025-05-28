"""Script for RAG evaluation."""
import evaluate
import faiss
import google.generativeai as genai
import pandas as pd
import time
from generate_response import generate_response, retrieve_context
from sentence_transformers import SentenceTransformer

# Global variables
PROCESSED_DATABASE_PATH = "../artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_KEY = "AIzaSyAV3VJG9STCErIBXz1LNls0V3SQ_UVi24U"

# Steup api key
genai.configure(api_key=GEMINI_API_KEY)

print("\nSetting up database, models and metrics...")
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
        reference = qa['response']

        print(f"# Question: {question}")
        print(f"# Reference: {reference}")

        # Retrieve context
        retrieved_context = retrieve_context(
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            processed_pages_df=processed_pages_df,
            question=question,
            k=10
        )

        # Generate response
        prediction = generate_response(
            gemini_chat=gemini_chat,
            question=question,
            retrieved_context=retrieved_context,
        )

        print(f"# Response: {prediction}\n")

        # Save reference and generated response
        references.append(reference)
        predictions.append(prediction)

        # Wait 6 seconds to avoid reach limit of 10 requests/min
        time.sleep(6.0)

    # Compute evaluation metrics
    # BLUE
    bleu = evaluate.load("bleu")
    bleu_score = float(
            bleu.compute(
            predictions=predictions, references=references
        )['bleu']
    )

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(
        predictions=predictions, references=references
    )
    rouge_score = float(
        (
            rouge_score['rouge1'] + rouge_score['rouge2'] +
            rouge_score['rougeL'] + rouge_score['rougeL']
        ) / 4
    )

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_score = float(
            meteor.compute(
            predictions=predictions, references=references
        )['meteor']
    )

    # Save metrics
    scores[subject] = {
        'bleu': bleu_score,
        'rouge': rouge_score,
        'meteor': meteor_score
    }
    print(f'# SCORE ({subject}):\n ## BLEU: {bleu_score}\n ## ROUGE: {rouge_score} \n ## METEOR: {meteor_score}')

score_df = pd.DataFrame(scores)
score_df.to_excel('scores.xlsx', index=False)

