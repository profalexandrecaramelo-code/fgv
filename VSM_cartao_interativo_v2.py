import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Introdução ao contexto
st.title("Exemplo de Aprendizado Supervisionado - Classificação com Naïve Bayes")

st.markdown("""
Este é um exemplo de **Aprendizado Supervisionado** utilizando o algoritmo **Naïve Bayes**.
No Aprendizado Supervisionado de **Classificação**, o modelo aprende a partir de dados rotulados para prever categorias (classes).
Aqui usamos o Naïve Bayes para classificar transações como **Fraude** ou **Normal** no contexto de operações com cartão de crédito.

Agora consideramos 5 fatores na descrição das transações:
- Local da transação
- Tipo de operação (saque, compra, pagamento)
- Valor (alto, baixo)
- Meio (online, presencial)
- Status inicial (aprovada, negada)
""")

# Dados fictícios ampliados
mensagens = [
    "compra aprovada supermercado presencial baixo",
    "saque não autorizado caixa eletrônico alto",
    "compra online aprovada ecommerce alto",
    "tentativa de saque suspeita exterior alto",
    "pagamento boleto aprovado online baixo",
    "compra negada cartão bloqueado presencial alto",
    "compra suspeita exterior online alto",
    "transação normal supermercado presencial baixo",
    "saque autorizado caixa eletrônico baixo",
    "compra exterior aprovada online alto",
    "compra aprovada loja presencial baixo",
    "pagamento autorizado online baixo",
    "saque aprovado caixa eletrônico baixo",
    "compra negada exterior online alto",
    "compra suspeita loja online alto",
    "pagamento suspeito boleto online alto",
    "compra aprovada supermercado presencial baixo",
    "saque suspeito caixa eletrônico alto",
    "compra exterior negada online alto",
    "compra normal loja presencial baixo",
    "pagamento normal boleto online baixo"
]

classes = [
    "normal",
    "fraude",
    "normal",
    "fraude",
    "normal",
    "fraude",
    "fraude",
    "normal",
    "normal",
    "fraude",
    "normal",
    "normal",
    "normal",
    "fraude",
    "fraude",
    "fraude",
    "normal",
    "fraude",
    "fraude",
    "normal",
    "normal"
]

df = pd.DataFrame({
    "mensagem": mensagens,
    "classe": classes
})

st.write("### Exemplo de dados")
st.dataframe(df)

# Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["mensagem"])
y = df["classe"]

# Modelo
modelo = MultinomialNB()
modelo.fit(X, y)

# Entrada do usuário
st.write("### Classifique sua transação")
entrada_usuario = st.text_input("Descreva a transação com os 5 fatores (ex: 'compra aprovada loja online baixo'):")

if st.button("Classificar"):
    if entrada_usuario.strip() == "":
        st.warning("Por favor, insira uma descrição para classificar.")
    else:
        X_novo = vectorizer.transform([entrada_usuario])
        predicao = modelo.predict(X_novo)[0]
        probas = modelo.predict_proba(X_novo)[0]
        st.success(f"O modelo prevê: **{predicao.upper()}**")
        st.write(f"Probabilidades: Normal = {probas[modelo.classes_ == 'normal'][0]:.2f}, Fraude = {probas[modelo.classes_ == 'fraude'][0]:.2f}")
