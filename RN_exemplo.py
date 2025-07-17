import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

st.set_page_config(page_title="Rede Neural - Aprovação de Crédito")

st.title("🏦 Simulador de Aprovação de Crédito com Rede Neural")
st.write("Este app usa uma rede neural com múltiplas camadas para prever se o crédito deve ser aprovado com base em dados do cliente.")

# ----- 1. Dados fictícios
data = {
    'pagou_antes':    [1, 1, 0, 1, 0, 0, 1, 0],
    'empregado':      [1, 0, 1, 1, 0, 1, 1, 0],
    'atrasou_contas': [0, 1, 1, 0, 1, 1, 0, 1],
    'aprovado':       [1, 0, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# ----- 2. Separar dados de entrada e saída
X = df[['pagou_antes', 'empregado', 'atrasou_contas']]
y = df['aprovado']

# ----- 3. Treinar rede neural com múltiplas camadas
model = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=1000, random_state=42)
model.fit(X, y)

# ----- 4. Interface interativa
st.header("🔍 Simule uma análise de crédito")

pagou_antes = st.selectbox("Já pagou dívidas anteriores?", ["Sim", "Não"])
empregado = st.selectbox("Está empregado?", ["Sim", "Não"])
atrasou = st.selectbox("Costuma atrasar contas?", ["Sim", "Não"])

# Converter entradas para binário
entrada = [
    1 if pagou_antes == "Sim" else 0,
    1 if empregado == "Sim" else 0,
    1 if atrasou == "Sim" else 0
]

# ----- 5. Prever resultado
if st.button("🔮 Ver resultado da análise"):
    previsao = model.predict([entrada])[0]
    prob = model.predict_proba([entrada])[0][1]

    if previsao == 1:
        st.success(f"✅ Crédito Aprovado! (Confiança: {prob:.2%})")
    else:
        st.error(f"❌ Crédito Negado. (Confiança: {prob:.2%})")

# ----- 6. Mostrar dados usados
with st.expander("📊 Ver dados de treinamento"):
    st.dataframe(df)
