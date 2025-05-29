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

print("\nSetting up database, models and metrics...\n")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
gemini_client = genai.GenerativeModel(model_name=GEMINI_MODEL)
gemini_chat = gemini_client.start_chat(history=[])

# Read QA benchmark data
benchmark_data = pd.read_json("../artifacts/benchmark.json")
benchmark_scores = {}
for subject in benchmark_data.columns:
    subject_qas = benchmark_data[subject]
    for qa in subject_qas:
        question = qa['question']
        reference = qa['response']
        reference_tokens = gemini_client.count_tokens(reference).total_tokens

        print(f"# Question: {question}")
        print(f"# Reference ({reference_tokens} tokens): {reference}")

        # Retrieve context
        retrieved_context = retrieve_context(
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            processed_pages_df=processed_pages_df,
            question=question,
            k=20
        )

        # Generate response
        prediction = generate_response(
            gemini_chat=gemini_chat,
            question=question,
            retrieved_context=retrieved_context,
            temperature=0.3,
        )
        prediction_tokens = gemini_client.count_tokens(prediction).total_tokens

        print(f"# Prediction ({prediction_tokens} tokens): {prediction}\n")

        # Wait 6 seconds to avoid reach limit of 10 requests/min
        time.sleep(6.0)

        # Compute evaluation metrics
        # BLUE
        bleu = evaluate.load("bleu")
        bleu_score = float(
                bleu.compute(
                predictions=[prediction], references=[reference]
            )['bleu']
        )

        # ROUGE
        rouge = evaluate.load("rouge")
        rouge_score = float(
            rouge.compute(
                predictions=[prediction], references=[reference]
            )['rougeL']
        )

        # METEOR
        meteor = evaluate.load("meteor")
        meteor_score = float(
                meteor.compute(
                predictions=[prediction], references=[reference]
            )['meteor']
        )

        # Save metrics
        benchmark_scores[subject] = {
            'question': question,
            'reference': reference,
            'prediction': prediction,
            'bleu': bleu_score,
            'rouge': rouge_score,
            'meteor': meteor_score
        }
        print(f" ## BLEU: {bleu_score}")
        print(f" ## ROUGE: {rouge_score}")
        print(f" ## METEOR: {meteor_score}")

benchmark_df = pd.DataFrame(benchmark_scores)
benchmark_df.to_excel('scores.xlsx', index=False)

