
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.title("🔍 Classificador de Crédito com VSM + SVM")

st.markdown("""
Este app demonstra o uso do **Vector Space Model (VSM)** e do **Support Vector Machine (SVM)**  
para classificar transações como **normal** ou **fraude**.

📌 **Entradas do modelo**
- **País:** Local da transação (ex.: Brasil, EUA, França).
- **Canal:** Se foi online ou presencial.
- **Tipo:** Tipo da operação (compra, saque, pagamento).
- **Descrição vetorial (VSM):** Texto formado por país + canal + tipo.
- **Valor (R$):** O valor numérico da transação, normalizado para o modelo.

📌 **Como funciona?**
- O **VSM** transforma as informações textuais em vetores numéricos.
- O **SVM** encontra um **hiperplano** que separa as classes no espaço vetorial.
- O hiperplano é a fronteira com maior margem entre transações normais e fraudulentas.

💡 **Aplicação no negócio:**  
Identificação automática de operações suspeitas, redução de fraudes e melhoria na segurança do cliente.
""")

# Dados fictícios ampliados
dados = {
    "transacao": [
        "Brasil online compra", "EUA presencial saque", "França online pagamento",
        "Brasil online saque", "EUA presencial compra", "França online compra",
        "Brasil presencial pagamento", "EUA online saque", "França presencial compra",
        "Brasil presencial saque", "EUA online pagamento", "França presencial saque",
        "Brasil presencial compra", "EUA online compra", "França presencial saque",
        "Brasil online pagamento", "EUA online saque", "França online saque",
        "Brasil presencial saque", "EUA presencial pagamento", "França presencial compra",
        "Brasil online compra", "EUA online pagamento", "França presencial pagamento"
    ],
    "valor": [
        500, 10000, 800, 15000, 300, 2000, 1200, 9000, 400, 11000,
        8500, 9500, 250, 700, 10500, 900, 9500, 9800, 12000, 500,
        350, 600, 8700, 1300
    ],
    "classe": [
        "normal", "fraude", "normal", "fraude", "normal", "normal",
        "normal", "fraude", "normal", "fraude",
        "fraude", "fraude", "normal", "normal", "fraude",
        "normal", "fraude", "fraude", "fraude", "normal",
        "normal", "normal", "fraude", "normal"
    ]
}
df = pd.DataFrame(dados)

# Pipeline
text_pipe = Pipeline([
    ('vectorizer', CountVectorizer())
])

preprocess = ColumnTransformer([
    ('text', text_pipe, 'transacao'),
    ('scaler', StandardScaler(), ['valor'])
])

modelo = make_pipeline(
    preprocess,
    SVC(kernel='linear', probability=True)
)

modelo.fit(df, df['classe'])

# Interface
st.write("### Insira os dados da transação:")

pais = st.selectbox("País", ["Brasil", "EUA", "França"])
canal = st.selectbox("Canal", ["online", "presencial"])
tipo = st.selectbox("Tipo", ["compra", "saque", "pagamento"])
valor = st.number_input("Valor da transação (R$)", min_value=0.0, value=1000.0)

entrada_df = pd.DataFrame({
    "transacao": [f"{pais} {canal} {tipo}"],
    "valor": [valor]
})

if st.button("Classificar"):
    resultado = modelo.predict(entrada_df)[0]
    prob = modelo.predict_proba(entrada_df)[0]
    prob_dict = dict(zip(modelo.classes_, prob))
    confianca = prob_dict[resultado]

    st.write(f"📝 Transação: **{pais} {canal} {tipo}, R$ {valor:,.2f}**")

    if resultado == "fraude":
        st.error(f"🚨 Esta transação foi classificada como **FRAUDE** com confiança: {confianca:.2%}")
    else:
        st.success(f"✅ Esta transação foi classificada como **NORMAL** com confiança: {confianca:.2%}")

    if 0.45 <= confianca <= 0.55:
        st.warning("⚠ **Atenção:** Esta transação está em região de ambiguidade próxima ao hiperplano. A revisão manual pode ser recomendada.")

st.markdown("""
### 📈 **Como o SVM funciona no VSM?**
- O **VSM** transforma o texto em um vetor no espaço das palavras.
- O **SVM** encontra um hiperplano no espaço vetorial (com o valor numérico incluído) que melhor separa as transações normais das fraudulentas.
- O hiperplano maximiza a margem entre os dois grupos, melhorando a generalização do modelo.
""")
