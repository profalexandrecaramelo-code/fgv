
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="CompreMais – Segmentação e Regras de Associação", layout="wide")
st.title("🛒 CompreMais – Segmentação de Clientes e Padrões de Compra")

st.sidebar.header("1. Upload dos Dados")
clientes_file = st.sidebar.file_uploader("Arquivo de clientes (clientes_compras.csv)", type=["csv"])
transacoes_file = st.sidebar.file_uploader("Arquivo de transações (transacoes_mercado.csv)", type=["csv"])

if clientes_file:
    st.header("🔹 Análise de Clusters de Clientes")
    df_clientes = pd.read_csv(clientes_file)

    st.subheader("Visualização dos dados de entrada")
    st.dataframe(df_clientes.head())

    numeric_cols = df_clientes.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) >= 2:
        st.markdown("### Selecione variáveis para clusterização")
        x_col = st.selectbox("Variável X", numeric_cols)
        y_col = st.selectbox("Variável Y", numeric_cols, index=1)
        n_clusters = st.slider("Número de clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_clientes['Cluster'] = kmeans.fit_predict(df_clientes[[x_col, y_col]])

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_clientes, x=x_col, y=y_col, hue='Cluster', palette='Set2', s=100, ax=ax)
        plt.title("Segmentação de Clientes por K-means")
        st.pyplot(fig)

        st.markdown("### Estatísticas por cluster")
        st.dataframe(df_clientes.groupby("Cluster")[numeric_cols].mean().round(2))
    else:
        st.warning("O dataset precisa de pelo menos duas colunas numéricas para clusterização.")

if transacoes_file:
    st.header("🔹 Regras de Associação com Apriori")
   transacoes = df_transacoes.apply(lambda row: [item for item in row if pd.notnull(item)], axis=1).tolist()
    st.subheader("Visualização das transações")
    st.dataframe(df_transacoes.head(10))

    st.markdown("### Parâmetros do Apriori")
    min_support = st.slider("Suporte mínimo", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Confiança mínima", 0.1, 1.0, 0.5, 0.05)

    transacoes = df_transacoes.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transacoes).transform(transacoes)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
    regras = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

    if not regras.empty:
        st.markdown("### Regras extraídas")
        st.dataframe(regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(2))
    else:
        st.warning("Nenhuma regra encontrada com os parâmetros escolhidos.")

st.sidebar.markdown("---")
st.sidebar.markdown("App desenvolvido para fins educacionais na disciplina de Cenários de IA – CompreMais ✨")
