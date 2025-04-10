
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori - Análise de Cesta de Compras", layout="centered")
st.title("🛒 Apriori: Regras de Associação")

st.markdown("""
Este app demonstra como aplicar o algoritmo **Apriori** para descobrir **regras de associação**
entre produtos em uma loja com base em transações de compra.
""")

# Dados simulados
st.subheader("📦 Dados de exemplo (transações)")
data = [
    ['leite', 'pão', 'manteiga'],
    ['leite', 'pão'],
    ['leite', 'café'],
    ['pão', 'manteiga'],
    ['leite', 'manteiga'],
    ['café', 'pão'],
    ['leite', 'pão', 'manteiga'],
]

df_exibicao = pd.DataFrame({'Transação': [i+1 for i in range(len(data))], 'Itens': [', '.join(t) for t in data]})
st.dataframe(df_exibicao)

# Pré-processamento
st.subheader("⚙️ Processamento e Geração de Regras")
produtos = sorted(set(item for trans in data for item in trans))
transacoes_codificadas = []
for trans in data:
    linha = {produto: (produto in trans) for produto in produtos}
    transacoes_codificadas.append(linha)

df = pd.DataFrame(transacoes_codificadas)

# Parâmetros
min_support = st.slider("Frequência mínima (support)", 0.1, 1.0, 0.3, 0.05)
min_confidence = st.slider("Confiança mínima (confidence)", 0.1, 1.0, 0.6, 0.05)

# Apriori
frequentes = apriori(df, min_support=min_support, use_colnames=True)
regras = association_rules(frequentes, metric="confidence", min_threshold=min_confidence)

# Exibir resultados
st.subheader("📈 Itens frequentes")
st.dataframe(frequentes)

st.subheader("🔗 Regras de associação geradas")
if regras.empty:
    st.warning("Nenhuma regra gerada com os parâmetros atuais.")
else:
    regras_view = regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    regras_view['antecedents'] = regras_view['antecedents'].apply(lambda x: ', '.join(list(x)))
    regras_view['consequents'] = regras_view['consequents'].apply(lambda x: ', '.join(list(x)))
    st.dataframe(regras_view)
