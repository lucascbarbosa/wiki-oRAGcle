"""evaluate_rag."""
import evaluate
import faiss
import pandas as pd
import torch
from generate_answer import generate_answer, retrieve_context
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "../artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
# LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# LLM_NAME = "allenai/OLMo-2-0425-1B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

# Setup variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch_dtype,
).to(device)
model = model.bfloat16().cuda()
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)


# Iterate each question and answer
qa_data = pd.read_json("../artifacts/questions_answers.json")
scores = {}
for subject in qa_data.columns:
    subject_qas = qa_data[subject]
    predictions = []
    references = []
    for qa in subject_qas:
        torch.cuda.empty_cache()
        question = qa['question']
        response_ref = qa['answer']
        retrieved_context = retrieve_context(
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            processed_pages_df=processed_pages_df,
            prompt=question,
            k=10
        )
        response_pred = generate_answer(
            device=device,
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            retrieved_context=retrieved_context,
            max_tokens=512,
        )
        print(
            f"Question: {question}\n"
            f"Reference: {response_ref}\n"
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
