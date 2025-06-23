
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Carrinho de Compras — Regras Interativas com Apriori")

st.markdown("""
## 📌 Case: Carrinho de Compras no Varejo

Este app identifica padrões de associação entre produtos no carrinho de compras.  
💡 O aluno escolhe itens e o app mostra o que costuma ser comprado junto com eles!
""")

# Carregar CSV
df_raw = pd.read_csv("carrinho_compras_200.csv")

# Preparar dados
produtos = ["leite", "pão", "manteiga", "queijo", "café", "arroz", "feijão", "macarrão", "refrigerante", "cerveja"]
df = pd.DataFrame(False, index=df_raw.index, columns=produtos)

for i, linha in df_raw.iterrows():
    itens = [x.strip() for x in linha["Transacao"].split(",")]
    for item in itens:
        if item in df.columns:
            df.at[i, item] = True

# Apriori
min_sup = st.slider("Escolha o suporte mínimo (%)", min_value=1, max_value=50, value=10) / 100
freq_itens = apriori(df, min_support=min_sup, use_colnames=True)
regras = association_rules(freq_itens, metric="confidence", min_threshold=0.5)

# Escolha dos itens pelo aluno
itens_escolhidos = st.multiselect("Escolha itens para verificar o que costuma ser comprado junto", produtos)

if itens_escolhidos:
    # Filtrar regras onde os itens escolhidos estão nos antecedentes
    regras_filtradas = regras[regras["antecedents"].apply(lambda x: set(itens_escolhidos).issubset(x))]
    if not regras_filtradas.empty:
        st.subheader("Regras encontradas")
        st.dataframe(regras_filtradas[["antecedents", "consequents", "support", "confidence", "lift"]])
    else:
        st.warning("Nenhuma regra encontrada para os itens selecionados com os parâmetros escolhidos.")
else:
    st.info("Selecione itens acima para ver as regras associadas.")

st.markdown("""
### 📌 Como funciona?
- O Apriori identifica combinações de produtos que aparecem juntos com frequência.
- Você seleciona os produtos e o app mostra o que mais costuma ser comprado junto com eles.
""")
