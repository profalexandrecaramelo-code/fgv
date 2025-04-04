
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
@st.cache_data
def carregar_dados():
    return pd.read_csv("dados_tickets_clusters.csv")

df = carregar_dados()

st.title("Análise de Clusters de Tickets de Suporte - K-means")

# Mostrar o DataFrame
st.subheader("Dados dos Tickets com Clusters")
st.dataframe(df)

# Selecionar colunas para visualização
st.subheader("Visualização Gráfica dos Clusters")
x_axis = st.selectbox("Escolha a variável do eixo X", df.columns[:-1], index=0)
y_axis = st.selectbox("Escolha a variável do eixo Y", df.columns[:-1], index=1)

# Gráfico de dispersão dos clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="cluster", palette="tab10", s=100)
plt.title("Clusters de Tickets de Suporte")
st.pyplot(plt)

# Estatísticas por cluster
st.subheader("Estatísticas por Cluster")
st.dataframe(df.groupby("cluster").mean(numeric_only=True))
