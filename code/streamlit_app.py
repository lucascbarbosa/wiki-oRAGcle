"""04__streamlit_app."""
import faiss
import pandas as pd
import streamlit as st
import torch
from generate_answer import generate_answer, retrieve_context
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "artifacts/faiss_index.index"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# LLM_NAME = "allenai/OLMo-2-0425-1B-Instruct"


# Setup llm varibles
if 'llm' not in st.session_state:
    print("\nSetting up llm variables...")
    # Setup torch configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
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
    st.session_state['llm'] = {
        'device': device,
        'embedding_model': embedding_model,
        'faiss_index': faiss_index,
        'processed_pages_df': processed_pages_df,
        'model': model,
        'tokenizer': tokenizer,
    }


# Interface Streamlit
st.markdown(
    "<h1>Ask the oRAGcle of <em>A Wiki of Ice and Fire</em></h1>",
    unsafe_allow_html=True
)

# Estado para armazenar o histórico de mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter question"):
    print(f"\nUser: {prompt}")

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # # Generate response
    print("\nGenerating response...")
    retrieved_context = retrieve_context(
        embedding_model=st.session_state['llm']['embedding_model'],
        faiss_index=st.session_state['llm']['faiss_index'],
        processed_pages_df=st.session_state['llm']['processed_pages_df'],
        prompt=prompt,
        k=30
    )
    response = generate_answer(
        device=st.session_state['llm']['device'],
        model=st.session_state['llm']['model'],
        tokenizer=st.session_state['llm']['tokenizer'],
        prompt=prompt,
        retrieved_context=retrieved_context,
        max_tokens=512,
    )

    print(f"\nAssistant: {response}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})