"""Script for RAG evaluation."""
import evaluate
import faiss
import google.generativeai as genai
import pandas as pd
import torch
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
# Database
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)

# Models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
gemini_client = genai.GenerativeModel(model_name=GEMINI_MODEL)

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bert = evaluate.load("bertscore")

# Read QA benchmark data
benchmark_data = pd.read_json("../artifacts/benchmark.json")
benchmark_scores = {}
for subject in benchmark_data.columns:
    subject_qas = benchmark_data[subject]
    for qa in subject_qas:
        # Start chat
        gemini_chat = gemini_client.start_chat(history=[])

        # Retrieve reference QA pair and difficulty level
        question = qa['question']
        reference = qa['response']
        difficulty = qa['difficulty']
        # Count reference tokens
        reference_tokens = gemini_client.count_tokens(reference).total_tokens

        print(f"# Question: {question}")
        print(f"# Reference ({reference_tokens} tokens): {reference}")

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
            temperature=0.3,
        )
        # Count prediction tokens
        prediction_tokens = gemini_client.count_tokens(prediction).total_tokens

        print(f"# Prediction ({prediction_tokens} tokens): {prediction}\n")

        # Compute evaluation metrics
        # BLUE
        bleu_score = float(
                bleu.compute(
                predictions=[prediction], references=[reference]
            )['bleu']
        )

        # ROUGE
        rouge_score = float(
            rouge.compute(
                predictions=[prediction], references=[reference]
            )['rougeL']
        )

        # METEOR
        meteor_score = float(
                meteor.compute(
                predictions=[prediction], references=[reference]
            )['meteor']
        )

        # BERTSCORE
        bert_score = bert.compute(
                predictions=[prediction],
                references=[reference],
                lang="en"
            )['f1']
        bert_score = sum(bert_score) / len(bert_score)

        # Save metrics
        benchmark_scores[subject] = {
            'subject': subject,
            'difficulty': difficulty,
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
        print(f" ## BERTSCORE: {bert_score}")

        del bert_score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


benchmark_df = pd.DataFrame(benchmark_scores)
benchmark_df.to_excel('scores.xlsx', index=False)
