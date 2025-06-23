
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

st.title("Classificação de Sementes - KNN Interativo com 3 Variáveis")

st.markdown("""
Neste exemplo de **Aprendizado Supervisionado**, utilizamos o algoritmo **K-Nearest Neighbors (KNN)**  
para classificar **sementes agrícolas** como **viáveis** ou **não viáveis** com base em:
- Peso da semente (gramas)
- Umidade da semente (%)
- Dureza da casca (N)

Você pode escolher o valor de K e observar como o modelo responde.
""")

# Gerando dados fictícios
np.random.seed(42)
n = 150
peso = np.random.uniform(5, 20, size=n)        # peso entre 5g e 20g
umidade = np.random.uniform(8, 20, size=n)     # umidade entre 8% e 20%
dureza = np.random.uniform(5, 20, size=n)      # dureza entre 5N e 20N

# Regra de viabilidade
viavel = ((peso > 10) & (umidade > 12) & (dureza > 10)).astype(int)

df = pd.DataFrame({
    "Peso (g)": peso,
    "Umidade (%)": umidade,
    "Dureza (N)": dureza,
    "Viável": viavel
})

st.write("### Exemplo dos dados")
st.dataframe(df.head())

# K escolhido pelo usuário
k = st.slider("Escolha o valor de K", min_value=1, max_value=15, value=5)

# Modelo
X = df[["Peso (g)", "Umidade (%)", "Dureza (N)"]]
y = df["Viável"]

modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X, y)

# Gráficos 2D (Peso vs Umidade)
st.write("### Distribuição (Peso x Umidade)")
fig1, ax1 = plt.subplots()
cores = ['red' if c == 0 else 'green' for c in y]
ax1.scatter(df["Peso (g)"], df["Umidade (%)"], c=cores, alpha=0.6)
ax1.set_xlabel("Peso (g)")
ax1.set_ylabel("Umidade (%)")
st.pyplot(fig1)

# Gráficos 2D (Peso vs Dureza)
st.write("### Distribuição (Peso x Dureza)")
fig2, ax2 = plt.subplots()
ax2.scatter(df["Peso (g)"], df["Dureza (N)"], c=cores, alpha=0.6)
ax2.set_xlabel("Peso (g)")
ax2.set_ylabel("Dureza (N)")
st.pyplot(fig2)

# Entrada do usuário
st.write("### Teste com nova semente")

peso_input = st.number_input("Peso da semente (g)", 5.0, 25.0, 10.0)
umidade_input = st.number_input("Umidade da semente (%)", 5.0, 25.0, 12.0)
dureza_input = st.number_input("Dureza da casca (N)", 5.0, 25.0, 10.0)

nova_amostra = np.array([[peso_input, umidade_input, dureza_input]])
pred = modelo.predict(nova_amostra)[0]

if pred == 1:
    st.success("🌱 Esta semente é classificada como **Viável**.")
else:
    st.error("❌ Esta semente é classificada como **Não Viável**.")
