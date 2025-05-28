"""Script for RAG evaluation."""
import faiss
import google.generativeai as genai
import pandas as pd
from generate_response import generate_response, retrieve_context
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    LLMContextRecall
)
from sentence_transformers import SentenceTransformer

# Global variables
PROCESSED_DATABASE_PATH = "../artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = "AIzaSyAV3VJG9STCErIBXz1LNls0V3SQ_UVi24U"

# Steup api key
genai.configure(api_key=GEMINI_API_KEY)

print("\nSetting up database, models and metrics...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
gemini_client = genai.GenerativeModel(model_name=GEMINI_MODEL)
gemini_chat = gemini_client.start_chat(history=[])
metrics = [
    AnswerCorrectness(),
    AnswerRelevancy(),
    AnswerSimilarity(),
    ContextPrecision(),
    ContextRecall(),
    Faithfulness(),
    LLMContextRecall()
]


# Read QA benchmark data
benchmark_data = pd.read_json("../artifacts/benchmark.json")
scores = {}
for subject in benchmark_data.columns:
    subject_qas = benchmark_data[subject]
    dataset = []
    for qa in subject_qas:
        question = qa['question']
        response_ref = qa['answer']

        print(f"Question: {question}")
        print(f"Reference: {response_ref}")

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

        print(f"Response: {response_pred}\n")

        dataset.append(
            {
                'user_input': question,
                'retrieved_contexts': retrieved_context,
                'response': response_pred,
                'reference': response_ref
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(gemini_client),
    )
    print(result)
    # # Evaluate metrics
    # scores[subject] = {
    #     'bleu': bleu_score,
    #     'rouge': float(rouge_score),
    # }
    # print(f'\n{subject}: {scores[subject]}\n')
