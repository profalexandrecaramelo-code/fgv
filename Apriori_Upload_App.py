
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Carrinho de Compras — Apriori com Upload de Base")

st.markdown("""
## 📌 Case: Carrinho de Compras no Varejo

Este app permite ao aluno enviar um arquivo CSV com transações.  
💡 O sistema aplica o **Apriori** e mostra os padrões encontrados para apoiar decisões de cross-sell e marketing.

### Formato esperado do CSV
- Uma coluna chamada **Transacao**.
- Os produtos da transação separados por vírgula.

Exemplo:
```
Transacao
pão, leite, café
sabão, shampoo
```
""")
# Upload do arquivo
arquivo = st.file_uploader("Envie o arquivo CSV com as transações", type="csv")

if arquivo is not None:
    df_raw = pd.read_csv(arquivo)

    if "Transacao" not in df_raw.columns:
        st.error("O arquivo precisa ter uma coluna chamada 'Transacao'.")
    else:
        produtos = sorted(set(", ".join(df_raw["Transacao"]).split(", ")))
        df = pd.DataFrame(False, index=df_raw.index, columns=produtos)

        for i, linha in df_raw.iterrows():
            itens = [x.strip() for x in linha["Transacao"].split(",")]
            for item in itens:
                if item in df.columns:
                    df.at[i, item] = True

        st.write("### Exemplo das transações processadas")
        st.dataframe(df.head())

        min_sup = st.slider("Escolha o suporte mínimo (%)", min_value=1, max_value=50, value=10) / 100
        freq_itens = apriori(df, min_support=min_sup, use_colnames=True)
        regras = association_rules(freq_itens, metric="confidence", min_threshold=0.5)

        st.subheader("Itens frequentes")
        st.dataframe(freq_itens)

        st.subheader("Regras de associação")
        if not regras.empty:
            st.dataframe(regras[["antecedents", "consequents", "support", "confidence", "lift"]])
        else:
            st.warning("Nenhuma regra encontrada com os parâmetros escolhidos.")

else:
    st.info("Por favor, envie um arquivo CSV para iniciar a análise.")

st.markdown("""
### 📌 Como funciona?
- O Apriori encontra combinações de produtos que aparecem juntas com frequência.
- A partir delas, cria regras SE ... ENTÃO ... com suporte, confiança e lift.
""")
