
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Segmentação de Clientes no Varejo com K-means (Base Externa)")

st.markdown("""
## 📌 Case: Segmentação de Clientes no Varejo

Uma rede varejista deseja segmentar seus clientes para aprimorar suas estratégias de marketing e fidelização.  
Para isso, utilizamos **K-means** para agrupar clientes com características de compra semelhantes.

### Dados usados (do arquivo CSV):
- **Valor gasto (R$)**: total gasto no último ano
- **Frequência de compras**: número de compras no último ano
- **Itens por compra**: média de itens adquiridos por compra

### Como o K-means funciona:
- O algoritmo cria **K grupos** de clientes com base nas variáveis selecionadas.
- Cada grupo é representado por um **centróide**.
- O número de grupos (**K**) é um **hiperparâmetro**, escolhido por você.

💡 O valor de **K** impacta diretamente a qualidade da segmentação.
""")

# Carregar o CSV
df = pd.read_csv("clientes_varejo.csv")
st.write("### Dados carregados")
st.dataframe(df.head())

# Escolha do K
k = st.slider("Escolha o número de clusters (K)", min_value=2, max_value=10, value=3)

# Rodar K-means
modelo = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = modelo.fit_predict(df)

# Visualização
st.subheader("Gráfico de dispersão dos clusters")

x_axis = st.selectbox("Variável do eixo X", df.columns[:-1], index=0)
y_axis = st.selectbox("Variável do eixo Y", df.columns[:-1], index=1)

fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="Cluster", palette="tab10", s=100, ax=ax)
plt.title("Clusters de clientes")
st.pyplot(fig)

# Estatísticas por cluster
st.subheader("Estatísticas por cluster")
st.dataframe(df.groupby("Cluster").mean())

st.markdown("""
### 📌 O que é o K no K-means?
- **K** é o número de clusters que o algoritmo deve gerar.
- É um **hiperparâmetro** porque você precisa definir antes de rodar o modelo.
- O K-means busca minimizar a variação dentro de cada grupo em relação ao seu centróide.
""")
