
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Análise de Carrinho de Compras com Apriori")

st.markdown("""
## 📌 Case: Carrinho de Compras no Varejo

Uma rede varejista quer descobrir padrões no comportamento de compra de seus clientes usando **Apriori**.  
Isso ajuda a responder perguntas como:
- Quais produtos costumam ser comprados juntos?
- Como sugerir produtos no e-commerce ou organizar o layout da loja?

### Dados usados:
Cada linha representa uma transação de um cliente.  
Os produtos estão indicados como colunas com valores **True/False**, indicando se o produto estava no carrinho.

💡 **Aplicação no negócio:** O varejo pode usar essas regras para cross-sell, promoções e recomendações.
""")

# Carregar o CSV
df_raw = pd.read_csv("carrinho_compras_200.csv")

# Pré-processamento: converter string para matriz booleana
# Separar produtos
produtos = ["leite", "pão", "manteiga", "queijo", "café", "arroz", "feijão", "macarrão", "refrigerante", "cerveja"]
df = pd.DataFrame(False, index=df_raw.index, columns=produtos)

for i, linha in df_raw.iterrows():
    itens = [x.strip() for x in linha["Transacao"].split(",")]
    for item in itens:
        if item in df.columns:
            df.at[i, item] = True

st.write("### Exemplo das transações processadas")
st.dataframe(df.head())

# min_support escolhido pelo usuário
min_sup = st.slider("Escolha o suporte mínimo (%)", min_value=1, max_value=50, value=10) / 100

# Rodar Apriori
freq_itens = apriori(df, min_support=min_sup, use_colnames=True)
regras = association_rules(freq_itens, metric="confidence", min_threshold=0.5)

# Mostrar resultados
st.subheader("Itens frequentes")
st.dataframe(freq_itens)

st.subheader("Regras de associação")
if not regras.empty:
    st.dataframe(regras[["antecedents", "consequents", "support", "confidence", "lift"]])
else:
    st.warning("Nenhuma regra encontrada com os parâmetros escolhidos.")

st.markdown("""
### 📌 O que faz o algoritmo Apriori?
- Encontra **conjuntos de itens frequentes** (que aparecem juntos em muitas transações).
- A partir deles, gera **regras do tipo SE ... ENTÃO ...** com medidas como suporte, confiança e lift.

💡 **Dica:** Tente diferentes valores de suporte mínimo e veja como as regras mudam.
""")
