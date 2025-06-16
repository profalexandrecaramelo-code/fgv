import streamlit as st
import random

# Dicionário de padrões e respostas
respostas_eliza = {
    "oi": ["Olá! Como você está se sentindo hoje?", "Oi! Em que posso te ajudar?"],
    "olá": ["Olá! Conte-me mais sobre isso.", "Oi! Como posso ajudar?"],
    "estou triste": ["Sinto muito ouvir isso. O que aconteceu?", "Quer falar mais sobre o motivo da tristeza?"],
    "estou feliz": ["Que ótimo! O que te deixou feliz?", "Fico feliz em ouvir isso!"],
    "não sei": ["Tudo bem não saber. Podemos explorar juntos?", "Por que você acha que não sabe?"],
    "sim": ["Entendo. Pode explicar melhor?", "Certo, pode me contar mais?"],
    "não": ["Por que não?", "Tem certeza? Vamos refletir sobre isso."],
    "adeus": ["Foi bom conversar com você. Até logo!", "Tchau! Espero ter ajudado."],
}

def eliza_responde(entrada):
    entrada = entrada.lower()
    for chave in respostas_eliza:
        if chave in entrada:
            return random.choice(respostas_eliza[chave])
    return "Conte-me mais sobre isso."

# Interface Streamlit
st.title("Chatbot ELIZA em Português")
st.write("Converse com o ELIZA, o assistente virtual em português.")

entrada_usuario = st.text_input("Você:", "")

if entrada_usuario:
    resposta = eliza_responde(entrada_usuario)
    st.text_area("ELIZA:", resposta, height=100)
