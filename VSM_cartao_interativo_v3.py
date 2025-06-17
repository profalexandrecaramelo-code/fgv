import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Introdução ao contexto
st.title("Exemplo de Aprendizado Supervisionado - Classificação de Crédito com VSM")

st.markdown("""
Este é um exemplo de **Aprendizado Supervisionado** com uso do **Vector Space Model (VSM)** para classificação de crédito.
No aprendizado supervisionado de **classificação**, o modelo aprende a partir de dados rotulados (fraude ou normal) e prevê a categoria de novas operações.
Aqui, cada transação é descrita por 4 fatores:
- **País** (ex: Brasil, EUA, França)
- **Valor** (baixo, médio, alto)
- **Canal** (online, presencial)
- **Tipo** (compra, saque, pagamento)

Esses fatores são transformados em vetores numéricos para o modelo.
""")

# Dados fictícios
dados = {
    "transacao": [
        "Brasil baixo online compra",
        "EUA alto presencial saque",
        "França médio online pagamento",
        "Brasil alto online saque",
        "EUA baixo presencial compra",
        "França alto online compra",
        "Brasil médio presencial pagamento",
        "EUA médio online saque",
        "França baixo presencial compra",
        "Brasil alto presencial saque",
        "EUA alto online pagamento",
        "França médio presencial saque",
        "Brasil baixo presencial compra",
        "EUA médio online compra",
        "França alto presencial saque"
    ],
    "classe": [
        "normal",
        "fraude",
        "normal",
        "fraude",
        "normal",
        "fraude",
        "normal",
        "fraude",
        "normal",
        "fraude",
        "fraude",
        "fraude",
        "normal",
        "normal",
        "fraude"
    ]
}

df = pd.DataFrame(dados)
st.write("### Exemplo de dados")
st.dataframe(df)

# Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["transacao"])
y = df["classe"]

# Treinamento
modelo = MultinomialNB()
modelo.fit(X, y)

# Entrada do usuário
st.write("### Simule sua transação")
pais = st.selectbox("País", ["Brasil", "EUA", "França"])
valor = st.selectbox("Valor", ["baixo", "médio", "alto"])
canal = st.selectbox("Canal", ["online", "presencial"])
tipo = st.selectbox("Tipo", ["compra", "saque", "pagamento"])

if st.button("Classificar transação"):
    descricao = f"{pais} {valor} {canal} {tipo}"
    X_novo = vectorizer.transform([descricao])
    predicao = modelo.predict(X_novo)[0]
    probas = modelo.predict_proba(X_novo)[0]
    st.success(f"O modelo prevê: **{predicao.upper()}**")
    st.write(f"Probabilidades: Normal = {probas[modelo.classes_ == 'normal'][0]:.2f}, Fraude = {probas[modelo.classes_ == 'fraude'][0]:.2f}")
