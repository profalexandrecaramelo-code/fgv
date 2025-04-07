import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Base de dados com duas variáveis
dados = pd.DataFrame({
    'investimento': [10, 15, 20, 25, 30, 18, 28, 10, 12, 22],
    'duracao': [7, 10, 15, 14, 20, 12, 18, 5, 10, 12],
    'agencia':['Blue', 'Blue', 'VM', 'VM', 'Blue', 'VM', 'Blue', 'Blue', 'Blue','Blue'],
    'receita': [50, 65, 55, 58, 120, 45, 110, 35, 45, 80]
})

# Variáveis independentes e alvo
X = dados[['investimento', 'duracao', 'agencia']]
y = dados['receita']

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Interface Streamlit
st.title("📈 Previsão de Receita com Regressão Linear")
st.write("Este app prevê a receita com base no investimento em marketing e duração da campanha.")

# Entradas do usuário
investimento = st.number_input("Investimento em marketing (R$ mil)", min_value=1, max_value=100, value=20)
duracao = st.number_input("Duração da campanha (dias)", min_value=1, max_value=60, value=10)
agencia = st.string_input("Agência Escolhida", Blue, VM)

# Prever receita
if st.button("🔍 Prever Receita"):
    novo_exemplo = pd.DataFrame({'investimento': [investimento], 'duracao': [duracao], 'agencia':[agencia] })
    previsto = modelo.predict(novo_exemplo)
    st.success(f"Receita prevista: R$ {previsto[0]:.2f} mil")

# Mostrar coeficientes
st.subheader("🧮 Detalhes do Modelo")
st.write(f"Intercepto (β0): {modelo.intercept_:.2f}")
st.write(f"Coeficiente β1 (Investimento): {modelo.coef_[0]:.2f}")
st.write(f"Coeficiente β2 (Duração): {modelo.coef_[1]:.2f}")
st.write(f"Coeficiente β3 (Agência): {modelo.coef_[2]:.2f}")
