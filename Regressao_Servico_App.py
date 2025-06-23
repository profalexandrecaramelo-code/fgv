
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Título e explicação inicial
st.title("Regressão Linear: Previsão de Preço de Serviço Profissional")

st.markdown("""
Este app demonstra o uso de **Regressão Linear** em **Aprendizado Supervisionado**.  
Estamos simulando um cenário onde uma empresa deseja **estimar o preço de um serviço profissional**
com base em:
- Idade do profissional
- Salário atual do profissional
- Anos de experiência
- Número de cursos de especialização

O objetivo é mostrar como essas variáveis impactam o preço estimado do serviço.
""")

# Gerando dados fictícios
np.random.seed(42)
n = 100
idade = np.random.randint(18, 65, size=n)
salario = np.random.randint(2000, 15000, size=n)
anos_experiencia = np.random.randint(0, 40, size=n)
cursos_especializacao = np.random.randint(0, 5, size=n)

preco_servico = (
    500 +
    (idade * 10) +
    (salario * 0.3) +
    (anos_experiencia * 50) +
    (cursos_especializacao * 200) +
    np.random.normal(0, 500, size=n)
)

df = pd.DataFrame({
    "Idade": idade,
    "Salário": salario,
    "Anos de Experiência": anos_experiencia,
    "Cursos de Especialização": cursos_especializacao,
    "Preço do Serviço (R$)": preco_servico
})

st.write("### Exemplo dos dados")
st.dataframe(df.head())

# Modelo
X = df[["Idade", "Salário", "Anos de Experiência", "Cursos de Especialização"]]
y = df["Preço do Serviço (R$)"]

modelo = LinearRegression()
modelo.fit(X, y)

st.write("### Coeficientes do Modelo")
coef_df = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
})
st.dataframe(coef_df)

# Gráficos
st.write("### Gráficos de Relação entre Variáveis e Preço do Serviço")

for coluna in X.columns:
    fig, ax = plt.subplots()
    ax.scatter(df[coluna], y, alpha=0.7)
    ax.set_xlabel(coluna)
    ax.set_ylabel("Preço do Serviço (R$)")
    ax.set_title(f"Preço do Serviço vs {coluna}")
    st.pyplot(fig)

# Previsão com dados do usuário
st.write("### Faça sua previsão")

idade_input = st.number_input("Idade", 18, 100, 30)
salario_input = st.number_input("Salário atual (R$)", 1000, 50000, 5000)
experiencia_input = st.number_input("Anos de experiência", 0, 50, 5)
cursos_input = st.number_input("Cursos de especialização", 0, 10, 1)

dados_usuario = np.array([[idade_input, salario_input, experiencia_input, cursos_input]])
previsao = modelo.predict(dados_usuario)[0]

st.success(f"Preço estimado do serviço: R$ {previsao:,.2f}")
