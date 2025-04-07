import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Base de dados
dados = pd.DataFrame({
    'investimento': [10, 15, 20, 25, 30, 18, 28, 10, 12, 22],
    'duracao': [7, 10, 15, 14, 20, 12, 18, 5, 10, 12],
    'agencia': ['Blue', 'Blue', 'VM', 'VM', 'Blue', 'VM', 'Blue', 'Blue', 'Blue', 'Blue'],
    'receita': [50, 65, 55, 58, 120, 45, 110, 35, 45, 80]
})

# One-hot encoding da variável categórica 'agencia'
dados_encoded = pd.get_dummies(dados, columns=['agencia'], drop_first=True)

# Separar variáveis independentes e alvo
X = dados_encoded.drop('receita', axis=1)
y = dados_encoded['receita']

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Interface no Streamlit
st.title("📈 Previsão de Receita com Regressão Linear")
st.write("Este app prevê a receita com base no investimento em marketing, duração da campanha e agência escolhida.")

# Entradas do usuário
investimento = st.number_input("Investimento em marketing (R$ mil)", min_value=1, max_value=100, value=20)
duracao = st.number_input("Duração da campanha (dias)", min_value=1, max_value=60, value=10)
agencia = st.selectbox("Agência escolhida", ["Blue", "VM"])

# Codificar a nova entrada como no treino
if st.button("🔍 Prever Receita"):
    # Manualmente codificando a agência (como no treino com drop_first=True)
    agencia_vm = 1 if agencia == "VM" else 0

    novo_exemplo = pd.DataFrame({
        'investimento': [investimento],
        'duracao': [duracao],
        'agencia_VM': [agencia_vm]
    })

    # Previsão
    previsto = modelo.predict(novo_exemplo)
    st.success(f"Receita prevista: R$ {previsto[0]:.2f} mil")

# Mostrar coeficientes
st.subheader("🧮 Detalhes do Modelo")
st.write(f"Intercepto (β0): {modelo.intercept_:.2f}")
for nome, coef in zip(X.columns, modelo.coef_):
    st.write(f"Coeficiente de {nome}: {coef:.2f}")
