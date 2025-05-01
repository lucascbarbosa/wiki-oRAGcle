"""04__streamlit_app."""
import streamlit as st
from generate_answer import generate_answer, retrieve_context


#############
# Functions #
def display_messages():
    """Display message in streamlit app."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                '<div style="text-align:left;"><span style="background-color:'
                '#DCF8C6; padding:10px; border-radius:10px; max-width: 80%;">'
                f'{msg["content"]}</span></div>',
                unsafe_allow_html=True
            )
        elif msg["role"] == "bot":
            st.markdown(
                '<div style="text-align:right;"><span style="background-color:'
                '#ECE5DD; padding:10px; border-radius:10px; max-width: 80%;">'
                f'{msg["content"]}</span></div>',
                unsafe_allow_html=True
            )


# Interface Streamlit
st.title("Chatbot de RAG - Interaja com o Modelo")

# Estado para armazenar o histórico de mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Exibir o histórico de mensagens
display_messages()

# Receber a entrada do usuário
user_input = st.text_input("Digite sua mensagem:")

if user_input:
    # Adicionar a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input})

    retrieved_context = retrieve_context(user_input, 5)

    # Gerar a resposta do chatbot
    bot_response = generate_answer(user_input)

    # Adicionar a resposta do bot ao histórico
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Exibir as novas mensagens
    display_messages()

    # Limpar o campo de entrada
    st.text_input("Digite sua mensagem:", "", key="input")
