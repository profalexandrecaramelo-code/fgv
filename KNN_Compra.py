
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Base de dados fictícia
dados = pd.DataFrame({
    'tempo': [12, 8, 15, 6, 14, 9],
    'paginas': [5, 3, 7, 2, 6, 4],
    'marketing': [1, 0, 1, 0, 1, 0],
    'comprou': [1, 0, 1, 0, 1, 0]
})

# Separando variáveis
X = dados[['tempo', 'paginas', 'marketing']]
y = dados['comprou']

# Modelo KNN
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)

# Interface Streamlit
st.title("🛒 Previsão de Compra com KNN")
st.write("Preveja se um cliente irá comprar com base em seu comportamento no site.")

# Entradas do usuário
tempo = st.slider("Tempo no site (min)", 1, 30, 10)
paginas = st.slider("Número de páginas visitadas", 1, 10, 4)
marketing = st.radio("Veio de campanha de marketing?", ["Sim", "Não"])
marketing_bin = 1 if marketing == "Sim" else 0

# Botão de previsão
if st.button("🔍 Prever"):
    novo_cliente = pd.DataFrame({
        'tempo': [tempo],
        'paginas': [paginas],
        'marketing': [marketing_bin]
    })
    previsao = modelo.predict(novo_cliente)[0]
    if previsao == 1:
        st.success("✅ O modelo prevê que o cliente **compraria** o produto.")
    else:
        st.error("❌ O modelo prevê que o cliente **não compraria** o produto.")
