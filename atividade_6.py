import streamlit as st
import pandas as pd
import numpy as np
import re
import random
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori - Mercado de Bairro (Simples)", layout="wide")
st.title("🛒 Apriori Simples — Treino e Predição")

st.markdown("""
**Fluxo:**  
1) Carregue a **base histórica** (Treino).  
2) Carregue a **base atual** (Predição).  
3) Ajuste o **suporte mínimo**.  
4) Encontre **ao menos 2 regras úteis** e explique por que geram valor (cross‑sell, layout, etc.).  
""")

# ------------------------- Gerador de bases de exemplo -------------------------
def gerar_transacoes_sinteticas(n, seed=42):
    random.seed(seed); np.random.seed(seed)
    lacteos = ["leite", "queijo", "manteiga", "iogurte"]
    panificados = ["pão", "biscoito", "bolo"]
    cafe_e_afins = ["café", "açúcar", "filtro de café"]
    massas = ["macarrão", "molho de tomate", "queijo ralado"]
    basicos = ["arroz", "feijão", "óleo", "farinha", "sal"]
    frios = ["presunto", "mussarela"]
    limpeza = ["detergente", "esponja", "sabão em pó", "amaciante"]
    higiene = ["papel higiênico", "sabonete", "creme dental", "escova de dente"]
    bebidas = ["refrigerante", "suco", "água mineral"]
    snacks = ["batata chips", "chocolate", "bala"]
    base_pop = lacteos + panificados + cafe_e_afins + massas + basicos + frios + limpeza + higiene + bebidas + snacks

    transacoes = []
    for _ in range(n):
        cesta = set()
        tam = np.random.choice([2,3,4,5,6], p=[0.15,0.30,0.30,0.18,0.07])
        if random.random() < 0.35:
            cesta.add("leite")
            if random.random() < 0.60: cesta.add("pão")
            if random.random() < 0.40: cesta.add("manteiga")
        if random.random() < 0.30:
            cesta.add("café")
            if random.random() < 0.70: cesta.add("açúcar")
            if random.random() < 0.35: cesta.add("filtro de café")
        if random.random() < 0.28:
            cesta.add("macarrão")
            if random.random() < 0.70: cesta.add("molho de tomate")
            if random.random() < 0.50: cesta.add("queijo ralado")
        if random.random() < 0.40:
            cesta.add("arroz")
            if random.random() < 0.75: cesta.add("feijão")
            if random.random() < 0.45: cesta.add("óleo")
        if random.random() < 0.25:
            cesta.add("detergente")
            if random.random() < 0.50: cesta.add("esponja")
        if random.random() < 0.25:
            cesta.add("papel higiênico")
            if random.random() < 0.40: cesta.add("sabonete")
        while len(cesta) < tam:
            cesta.add(random.choice(base_pop))
        transacoes.append(", ".join(sorted(cesta)))
    return pd.DataFrame({"Transacao": transacoes})

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    if st.button("🧪 Baixar base de exemplo — TREINO (1000)"):
        df = gerar_transacoes_sinteticas(1000, seed=42)
        st.download_button("📥 Download treino_1000.csv", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="treino_1000.csv", mime="text/csv")
with col_ex2:
    if st.button("🧪 Baixar base de exemplo — PREDIÇÃO (100)"):
        df = gerar_transacoes_sinteticas(100, seed=777)
        st.download_button("📥 Download predicao_100.csv", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="predicao_100.csv", mime="text/csv")

# ------------------------- Helpers -------------------------
def split_itens(x: str):
    if pd.isna(x):
        return []
    partes = re.split(r"[;,]", str(x))
    return [p.strip().lower() for p in partes if p and p.strip()]

def listas_para_ohe(series_listas):
    todos = sorted({it for lst in series_listas for it in lst})
    ohe = pd.DataFrame(False, index=range(len(series_listas)), columns=todos)
    for i, lst in enumerate(series_listas):
        for it in lst:
            if it in ohe.columns:
                ohe.at[i, it] = True
    return ohe

# ------------------------- Uploads -------------------------
st.sidebar.header("📂 Upload dos Arquivos (CSV com coluna 'Transacao')")
hist_file = st.sidebar.file_uploader("Base Histórica (Treino)", type=["csv"])
pred_file = st.sidebar.file_uploader("Base Atual (Predição)", type=["csv"])

# Parâmetro — manter simples (apenas suporte)
min_sup = st.sidebar.slider("Suporte mínimo (%)", 1, 50, 10) / 100.0

# ------------------------- Treino -------------------------
if hist_file is not None:
    st.subheader("📊 Treino — Base Histórica")
    df_hist = pd.read_csv(hist_file)
    if "Transacao" not in df_hist.columns:
        st.error("A base de treino precisa ter a coluna 'Transacao'.")
        st.stop()

    listas = df_hist["Transacao"].apply(split_itens)
    ohe = listas_para_ohe(listas)
    st.caption("Amostra (one‑hot das transações)")
    st.dataframe(ohe.head())

    # Apriori + Regras
    freq = apriori(ohe, min_support=min_sup, use_colnames=True)
    if freq.empty:
        st.warning("Nenhum item frequente encontrado. Reduza o suporte.")
        st.stop()

    regras = association_rules(freq, metric="confidence", min_threshold=0.5)
    if regras.empty:
        st.warning("Nenhuma regra encontrada com confiança ≥ 0.5. Tente reduzir o suporte.")
        st.stop()

    st.subheader("📜 Regras de Associação (Treino)")
    regras_view = (regras[["antecedents", "consequents", "support", "confidence", "lift"]]
                   .sort_values("lift", ascending=False)
                   .reset_index(drop=True))
    st.dataframe(regras_view, use_container_width=True)

    st.download_button("⬇️ Baixar Regras (CSV)",
        data=regras_view.to_csv(index=False).encode("utf-8"),
        file_name="regras_apriori_treino.csv",
        mime="text/csv"
    )

    # ------------------------- Predição -------------------------
    if pred_file is not None:
        st.subheader("🔮 Predição — Recomendações por Carrinho")
        df_pred = pd.read_csv(pred_file)
        if "Transacao" not in df_pred.columns:
            st.error("A base de predição precisa ter a coluna 'Transacao'.")
            st.stop()

        def recomendar(itens_str):
            itens = set(split_itens(itens_str))
            recs = set()
            for _, r in regras.iterrows():
                ante = r["antecedents"]; cons = r["consequents"]
                if ante.issubset(itens):
                    recs |= (cons - itens)  # só recomenda o que não está no carrinho
            return ", ".join(sorted(recs)) if recs else ""

        df_pred_out = df_pred.copy()
        df_pred_out["Sugestoes"] = df_pred_out["Transacao"].apply(recomendar)

        st.dataframe(df_pred_out, use_container_width=True)
        st.download_button("⬇️ Baixar Recomendações (CSV)",
            data=df_pred_out.to_csv(index=False).encode("utf-8"),
            file_name="recomendacoes_predicao.csv",
            mime="text/csv"
        )
else:
    st.info("Carregue a base histórica (Treino) e a base atual (Predição). As bases devem conter a coluna 'Transacao'.")
