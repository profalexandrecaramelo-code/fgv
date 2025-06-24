
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Carrinho de Compras — Apriori com Upload de Base")

st.markdown("""
## 📌 Case: Mercado de Bairro e o uso da IA

Você é um **consultor de IA** contratado por um **dono de mercado de bairro**.  
Esse dono quer entender os padrões de compra dos seus clientes para melhorar as ofertas, organizar melhor o layout da loja e otimizar o estoque.

➡ O consultor, após analisar o problema de negócio e os dados, recomendou o uso do **algoritmo Apriori** para identificar associações entre produtos no carrinho de compras.

💡 **Seu papel como aluno:** Analise os padrões encontrados e recomende ações ao dono do mercado.
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
## 🎯 Atividade proposta

✅ Carregue a base de dados com as transações do mercado.  
✅ Ajuste o suporte mínimo para gerar regras significativas.  
✅ Identifique **ao menos 2 regras úteis** que você recomendaria ao dono do mercado.  
✅ Explique por que essas regras podem gerar valor (ex.: promoções cruzadas, reorganização da loja).

💡 Dica: Foque em regras com bom **lift** e **confiança**.
""")
