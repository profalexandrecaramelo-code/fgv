import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Introdução ao contexto
st.title("Exemplo de Aprendizado Supervisionado - Classificação com KNN")

st.markdown("""
Este é um exemplo de **Aprendizado Supervisionado** utilizando o algoritmo **K-Nearest Neighbors (KNN)**.
No Aprendizado Supervisionado de Classificação, o modelo aprende a partir de dados rotulados para prever categorias (classes).
O KNN classifica novos dados com base nos exemplos mais próximos no conjunto de treino.

Aqui simulamos um cenário em que os dados determinam se um cliente comprou ou não um produto.
""")

# Gerando dados fictícios
np.random.seed(42)
n = 100
idade = np.random.randint(18, 70, size=n)
salario = np.random.randint(1000, 20000, size=n)
comprou = (salario > 10000).astype(int)

# Criando DataFrame
df = pd.DataFrame({
    "idade": idade,
    "salario": salario,
    "comprou": comprou
})

st.write("### Exemplo de dados")
st.dataframe(df.head())

# Modelo
X = df[["idade", "salario"]]
y = df["comprou"]

modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X, y)

# Visualização
st.write("### Gráfico: Idade x Salário (Compra = Sim/Não)")
plt.figure(figsize=(8,6))
cores = ['red' if c == 0 else 'green' for c in y]
plt.scatter(df["idade"], df["salario"], c=cores, alpha=0.6)
plt.xlabel("Idade")
plt.ylabel("Salário")
plt.title("Distribuição dos dados")
st.pyplot(plt)

# Entrada do usuário
st.write("### Previsão com seus próprios dados")
idade_in = st.number_input("Idade", min_value=18, max_value=100, value=30)
salario_in = st.number_input("Salário", min_value=0, value=5000)

if st.button("Verificar se compraria"):
    dados_input = pd.DataFrame({
        "idade": [idade_in],
        "salario": [salario_in]
    })
    resultado = modelo.predict(dados_input)[0]
    if resultado == 1:
        st.success("O modelo prevê que o cliente **compraria** o produto.")
    else:
        st.info("O modelo prevê que o cliente **não compraria** o produto.")
