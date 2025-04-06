import streamlit as st
import re

st.set_page_config(page_title="Chatbot ELIZA em Português", page_icon="💬")

st.title("🤖 Chatbot ELIZA (em Português)")
st.markdown("Simulação do ELIZA original, com regras simples de resposta.")

def responder(usuario_input):
    respostas = [
        (r"(.*) meu nome é (.*)", "Olá, {1}! Como posso ajudá-lo(a) hoje?"),
        (r"(oi|olá|bom dia|boa tarde|boa noite)", "Olá! Como você está se sentindo hoje?"),
        (r"(.*) estou (.*)", "Por que você está {1}?"),
        (r"(.*) estou triste(.*)", "Sinto muito por isso. Quer falar sobre o que está te deixando triste?"),
        (r"(.*) quero (.*)", "Por que você quer {1}?"),
        (r"(.*) não consigo (.*)", "O que você acha que está te impedindo de {1}?"),
        (r"(.*) problemas (.*)", "Conte-me mais sobre esses problemas."),
        (r"(.*)", "Entendo... pode me explicar um pouco mais sobre isso?")
    ]

    for padrao, resposta in respostas:
        match = re.match(padrao, usuario_input.lower())
        if match:
            grupos = match.groups()
            for i, g in enumerate(grupos):
                resposta = resposta.replace(f"{{{i}}}", g)
            return resposta.capitalize()

    return "Pode me contar mais sobre isso?"

# Histórico de conversa
if "historico" not in st.session_state:
    st.session_state.historico = []

usuario_input = st.text_input("Você:", key="input")

if usuario_input:
    resposta = responder(usuario_input)
    st.session_state.historico.append(("Você", usuario_input))
    st.session_state.historico.append(("ELIZA", resposta))

# Mostrar conversa
for remetente, mensagem in st.session_state.historico:
    if remetente == "Você":
        st.markdown(f"**👤 {remetente}:** {mensagem}")
    else:
        st.markdown(f"**🤖 {remetente}:** {mensagem}")
