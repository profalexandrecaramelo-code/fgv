import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Introdução ao contexto
st.title("Exemplo de Aprendizado Supervisionado - Regressão Linear")

st.markdown("""
Este é um exemplo de **Aprendizado Supervisionado** utilizando **Regressão Linear**.
No Aprendizado Supervisionado, o modelo aprende a partir de dados rotulados, ou seja, dados onde sabemos o valor da variável alvo.
No caso da Regressão, o objetivo é prever um valor contínuo com base em uma ou mais variáveis independentes.

Aqui simulamos um cenário com múltiplas variáveis independentes que influenciam o valor alvo.
""")

# Gerando dados fictícios
np.random.seed(42)
n = 100
idade = np.random.randint(18, 65, size=n)
salario = np.random.randint(2000, 15000, size=n)
anos_experiencia = np.random.randint(0, 40, size=n)
cursos_especializacao = np.random.randint(0, 5, size=n)

# Valor alvo: preço estimado do serviço
preco_servico = 500 + (idade * 10) + (salario * 0.3) + (anos_experiencia * 50) + (cursos_especializacao * 200) + np.random.normal(0, 500, size=n)

# Criando DataFrame
df = pd.DataFrame({
    "idade": idade,
    "salario": salario,
    "anos_experiencia": anos_experiencia,
    "cursos_especializacao": cursos_especializacao,
    "preco_servico": preco_servico
})

st.write("### Exemplo de dados")
st.dataframe(df.head())

# Treinamento do modelo
X = df[["idade", "salario", "anos_experiencia", "cursos_especializacao"]]
y = df["preco_servico"]

modelo = LinearRegression()
modelo.fit(X, y)

st.write("### Coeficientes do Modelo")
coef_df = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
})
st.dataframe(coef_df)

st.write(f"Intercepto: {modelo.intercept_:.2f}")

# Predição
df["preco_previsto"] = modelo.predict(X)

# Visualização
st.write("### Gráfico: Preço Real x Preço Previsto")
plt.figure(figsize=(8,6))
plt.scatter(df["preco_servico"], df["preco_previsto"])
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
plt.title("Preço Real vs. Preço Previsto")
plt.plot([df["preco_servico"].min(), df["preco_servico"].max()],
         [df["preco_servico"].min(), df["preco_servico"].max()],
         color='red', linestyle='--')
st.pyplot(plt)

# Entrada de dados do usuário
st.write("### Previsão com seus próprios dados")
idade_in = st.number_input("Idade", min_value=18, max_value=100, value=30)
salario_in = st.number_input("Salário", min_value=0, value=5000)
anos_exp_in = st.number_input("Anos de experiência", min_value=0, max_value=50, value=5)
cursos_in = st.number_input("Cursos de especialização", min_value=0, max_value=10, value=1)

if st.button("Calcular preço estimado"):
    dados_input = pd.DataFrame({
        "idade": [idade_in],
        "salario": [salario_in],
        "anos_experiencia": [anos_exp_in],
        "cursos_especializacao": [cursos_in]
    })
    preco_estimado = modelo.predict(dados_input)[0]
    st.success(f"O preço estimado do serviço é: R$ {preco_estimado:.2f}")
