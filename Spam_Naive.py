
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Dados de treino (exemplo)
emails = [
    "Oferta especial! Compre agora",
    "Promoção exclusiva para você",
    "Última chance",
    "Grande oportunidade",
    "Oportunidade especial",
    "Basta clicar aqui",
    "Seu prêmio aguarda",
    "Documento solicitado",
    "Reunião amanhã às 10h",
    "Prezado colaborador",
    "Relatório financeiro do trimestre"
]
labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 1 = spam, 0 = não spam

# Treinando o modelo com pipeline
modelo = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
modelo.fit(emails, labels)

# Interface com Streamlit
st.title("🔍 Classificador de E-mails Spam")
st.write("Digite o conteúdo de um e-mail abaixo:")

entrada = st.text_area("Texto do e-mail")

if st.button("Classificar"):
    if entrada.strip() == "":
        st.warning("Por favor, digite um e-mail para classificar.")
    else:
        resultado = modelo.predict([entrada])[0]
        prob = modelo.predict_proba([entrada])[0][1]

        if resultado == 1:
            st.error(f"🚨 Este e-mail foi classificado como **SPAM** (confiança: {prob:.2%})")
        else:
            st.success(f"✅ Este e-mail **não é SPAM** (confiança: {prob:.2%})")
