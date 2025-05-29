"""Script for the chatbot web app."""
import faiss
import google.generativeai as genai
import pandas as pd
import streamlit as st
from generate_response import generate_response, retrieve_context
from sentence_transformers import SentenceTransformer

# Carrega bases e modelos
PROCESSED_DATABASE_PATH = "../artifacts/processed_database.parquet"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = "AIzaSyAV3VJG9STCErIBXz1LNls0V3SQ_UVi24U"

# Steup api key
genai.configure(api_key=GEMINI_API_KEY)


# Setup llm varibles
if 'llm' not in st.session_state:
    print("\nSetting up llm variables...")
    # Setup variables
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    processed_pages_df = pd.read_parquet(PROCESSED_DATABASE_PATH)
    gemini_client = genai.GenerativeModel(model_name=GEMINI_MODEL)
    gemini_chat = gemini_client.start_chat(history=[])
    st.session_state['llm'] = {
        'embedding_model': embedding_model,
        'faiss_index': faiss_index,
        'processed_pages_df': processed_pages_df,
        'gemini_chat': gemini_chat
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

if question := st.chat_input("Enter question"):
    print(f"\nUser: {question}")

    # Display user message in chat message container
    st.chat_message("user").markdown(question)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # # Generate response
    print("\nGenerating response...")
    retrieved_context = retrieve_context(
        embedding_model=st.session_state['llm']['embedding_model'],
        faiss_index=st.session_state['llm']['faiss_index'],
        processed_pages_df=st.session_state['llm']['processed_pages_df'],
        question=question,
        k=20
    )
    response = generate_response(
        gemini_chat=st.session_state['llm']['gemini_chat'],
        question=question,
        retrieved_context=retrieved_context,
    )

    print(f"\nAssistant: {response}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})