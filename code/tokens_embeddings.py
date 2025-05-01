"""02__tokens_embeddings."""
import faiss
import numpy as np
import pandas as pd
import pickle
import re
import tiktoken
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_SIZE = 400
OVERLAP = 50
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DATABASE_PATH = "artifacts/database.parquet"
PROCESSED_DATABASE_PATH = "artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "artifacts/faiss_index.index"
METADATA_PATH = "artifacts/metadata.pkl"


def clean_text(text: str) -> str:
    """Clean page content."""
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # Remove {{...}}
    text = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', text)          # [[link|text]] -> text
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)               # [[text]] -> text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)               # **bold** -> plain
    text = re.sub(r'thumb\|.*', '', text)                      # remove images
    text = re.sub(r'\n+', '\n', text)                          # multiple newlines
    return text.strip()


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap: int = OVERLAP) -> list:
    """Split pages content into chunks."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


def process_pages(pages_df: pd.DataFrame) -> pd.DataFrame:
    """Process pages content."""
    tiktoken.encoding_for_model("gpt-3.5-turbo")
    model = SentenceTransformer(EMBEDDING_MODEL, device='cuda')

    records = []
    print("2. Limpando e separando texto...")
    for _, row in tqdm(pages_df.iterrows(), total=len(pages_df)):
        cleaned = clean_text(row['content'])
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            records.append({
                'pageid': row['pageid'],
                'title': row['title'],
                'chunk_id': f"{row['pageid']}_{i}",
                'text': chunk
            })

    # Criar DataFrame de chunks
    chunks_df = pd.DataFrame(records)

    print("3. Gerando embeddings...")
    embeddings = model.encode(chunks_df['text'].tolist(), show_progress_bar=True)

    chunks_df['embedding'] = embeddings.tolist()
    return chunks_df


def save_faiss_index(chunks_df: pd.DataFrame):
    """Save embeddings and metadata."""
    # Converte embeddings para float32 numpy array
    embedding_matrix = np.vstack(chunks_df['embedding'].values).astype('float32')

    # Cria o índice FAISS
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 = distância euclidiana
    index.add(embedding_matrix)

    # Salva o índice FAISS em disco
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index salvo em: {FAISS_INDEX_PATH}")

    # Salva os metadados (sem os embeddings)
    metadata = chunks_df.drop(columns=["embedding"])
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadados salvos em: {METADATA_PATH}")


print("1. Carregando base de dados...")
pages_df = pd.read_parquet(DATABASE_PATH)
processed_chunks_df = process_pages(pages_df)
processed_chunks_df.to_parquet(PROCESSED_DATABASE_PATH, index=False)

print("4. Salvando FAISS index...")
save_faiss_index(processed_chunks_df)
