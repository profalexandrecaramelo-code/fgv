
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Interface Streamlit
st.title("Análise de Crédito - Rede Neural SEM camada oculta")

# Upload do CSV
uploaded_file = st.file_uploader("Faça upload do arquivo de dados (.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, envie o arquivo 'dados_credito.csv' para continuar.")
    st.stop()

# Separar entradas e saída
X = df.drop("aprovado", axis=1)
y = df["aprovado"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rede neural SEM camada oculta
model = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Avaliação
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.metric("Acurácia no conjunto de teste", f"{acc*100:.2f}%")

st.markdown("### Faça a simulação de um novo cliente")

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

st.subheader("Resultado da Análise do Novo Cliente")
st.metric("Probabilidade de Aprovação", f"{prob*100:.1f}%")

if pred == 1:
    st.success("Crédito Aprovado")
else:
    st.error("Crédito Negado")
