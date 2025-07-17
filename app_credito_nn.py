
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv("dados_credito.csv")
    return df

df = load_data()

# Separar entradas e saída
X = df.drop("aprovado", axis=1)
y = df["aprovado"]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar modelo
model = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Interface Streamlit
st.title("Análise de Crédito com Rede Neural")
st.markdown("Preencha os dados abaixo para saber se o crédito será aprovado.")

# Inputs do usuário
renda = st.slider("Renda mensal (R$)", 1000, 15000, 5000)
tempo = st.slider("Tempo de emprego (anos)", 0.0, 10.0, 3.0)
idade = st.slider("Idade", 18, 65, 35)
dividas = st.slider("Valor total de dívidas (R$)", 0, 15000, 2000)
score = st.slider("Score de crédito", 300, 850, 600)

# Preparar input
input_data = np.array([[renda, tempo, idade, dividas, score]])
input_scaled = scaler.transform(input_data)

# Previsão
prob = model.predict_proba(input_scaled)[0][1]
pred = model.predict(input_scaled)[0]

st.subheader("Resultado da Análise")
st.metric("Probabilidade de Aprovação", f"{prob*100:.1f}%")

if pred == 1:
    st.success("Crédito Aprovado")
else:
    st.error("Crédito Negado")
